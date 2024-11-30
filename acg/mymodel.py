from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import einops
import contextlib
from Qformer import BertConfig, BertLMHeadModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from typing import List
import pickle as pkl
import sys
import io

from config import config 


    
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
        

class MyModel(nn.Module):
    def __init__(self, max_frame_pos=128, 
                 window=30, num_query_tokens=32, 
                 num_video_query_token=32, num_features=1024, 
                 device = "cuda", inference=False):
        
        super().__init__()
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.language_model.tokenizer_name)
        self.tokenizer.add_tokens(["[PLAYER]","[TEAM]","([TEAM])"], special_tokens=True)
        self.llama_model = AutoModelForCausalLM.from_pretrained(config.model.language_model.llm_name, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.tokenizer))
        
        self.ln_vision = LayerNorm(num_features)
        self.num_query_tokens = num_query_tokens,
        self.num_video_query_token = num_video_query_token
        self.inference = inference
        
        
         # Initialize video Q-former
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,
                                                                             vision_width=num_features,
                                                                             num_hidden_layers =2)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # llama projection
        self.llama_proj = nn.Linear(
            self.video_Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        # video frame positional embedding
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, num_features)
        self.window = window

        # move to device
        self.llama_model = self.llama_model.to(self.device)
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        self.video_Qformer = self.video_Qformer.to(self.device)
        self.llama_proj = self.llama_proj.to(self.device)
        self.ln_vision = self.ln_vision.to(self.device)
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.video_frame_position_embedding = self.video_frame_position_embedding.to(self.device)

        
        
    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def forward(self, batch, validating=False):
        
        video_features = batch['vid_features'] #B,T,P,D or [T,P,D]
            
        input_ids= batch['input_ids'] #Bxmax(T)
        atts_llama = batch['attention_mask']  #Bxmax(T)
        targets= batch['labels'] #Bxmax(T)
        
        print(input_ids.shape, atts_llama.shape, targets.shape)
        try:
            print(video_features.shape)
        except:
            #temporary matching
            print(len(video_features))
            f = []
            for i in video_features:
                if i.size(0) >= 3:
                     f.append(i[:3]) 
            video_features = torch.stack(f).to(self.device)
            
            print(video_features.shape) #BxTxFxD
            
        batch_size, time_length, _, _ = video_features.size()
        
        video_features = self.ln_vision(video_features)
        
        video_features = einops.rearrange(video_features, 'b t n f -> (b t) n f', b=batch_size, t=time_length)
        position_ids = torch.arange(time_length, dtype=torch.long, device=video_features.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        frame_position_embeddings = self.video_frame_position_embedding(position_ids)
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        
        frame_hidden_state = einops.rearrange(video_features, '(b t) n f -> b t n f',b=batch_size,t=time_length)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state
        
        frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
        
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(frame_hidden_state)
        
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1).to(frame_hidden_state.device)
        
        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        
        video_hidden = video_query_output.last_hidden_state
        
        inputs_llama = self.llama_proj(video_hidden)
        
        if self.inference:
            return self.generate_text(inputs_llama)
        
        if validating:
            pass
        
        visual_label = torch.full((batch_size, self.num_video_query_token), -100, dtype=targets.dtype)
        concat_targets = torch.cat((visual_label, targets), dim=1).to(self.device)
        temp_input_ids = input_ids.clone().to(self.device)
        targets_embeds = self.llama_model.model.embed_tokens(temp_input_ids)
        embedding_cat = torch.cat((inputs_llama, targets_embeds), dim=1)
        mask_prefix = torch.ones(batch_size, self.num_video_query_token, dtype=atts_llama.dtype)
        mask = torch.concat((mask_prefix, atts_llama), dim=1).to(self.device)
    
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=embedding_cat,
                attention_mask=mask,
                return_dict=True,
                labels=concat_targets,
            )
        sys.stdout = original_stdout
        loss = outputs.loss
        return loss

        
    def generate_text(self, inputs_llama):
        start_embeds = self.llama_model.model.embed_tokens(torch.tensor([128000]).to(self.device))
        inputs_llama_with_s = torch.cat([inputs_llama, start_embeds.expand(inputs_llama.size(0), -1, -1)], dim=1).to(dtype=torch.bfloat16)
        temp_res_tokens = self.llama_model.generate(
            logits_processor=self.logits_prosessors,
            renormalize_logits=True,
            inputs_embeds=inputs_llama_with_s,
            max_new_tokens=128,
            num_beams=5,
            do_sample=True,
            min_length=5,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )
        res_text = process_output_tokens(self, temp_res_tokens)
        return res_text


        
        
        
    
    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
        
        def generate_text(self, inputs_llama):
            start_embeds = self.llama_model.model.embed_tokens(torch.tensor([128000]).to(self.device))
            inputs_llama_with_s = torch.cat([inputs_llama, start_embeds.expand(inputs_llama.size(0), -1, -1)], dim=1).to(dtype=torch.bfloat16)
            temp_res_tokens = self.llama_model.generate(
                logits_processor=self.logits_prosessors,
                renormalize_logits=True,
                inputs_embeds=inputs_llama_with_s,
                max_new_tokens=128,
                num_beams=5,
                do_sample=True,
                min_length=5,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=1,
                temperature=1.0,
            )
            res_text = process_output_tokens(self, temp_res_tokens)
            return res_text
    
    

        
        
from dataset import ACGDataset

ds = ACGDataset(config.data.train_path)

batch = [ds[0], ds[1], ds[3], ds[4]]

collated_batch = ds.collator(batch)


model  = MyModel()

model(collated_batch)
        