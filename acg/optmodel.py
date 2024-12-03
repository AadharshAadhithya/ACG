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
import numpy as np

from config import config 
from transformers import BertTokenizer


def process_output_tokens(predict_model, tokens):
    output_texts = []
    for output_token in tokens:
        output_text = predict_model.tokenizer.decode(output_token)
        end_token_index = output_text.find('<|end_of_text|>')
        if end_token_index != -1:
            output_text = output_text[:end_token_index]
        output_texts.append(output_text)
    return output_texts
    
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
        

class OPTModel(nn.Module):
    def __init__(self,opt_model, opt_tokenizer, max_frame_pos=128, 
                 window=30, num_query_tokens=32, 
                 num_video_query_token=32, num_features=1024, 
                 device = "cuda", inference=False):
        
        super().__init__()
        
        self.device = device
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_tokenizer)
        #self.opt_tokenizer.add_tokens(["[PLAYER]","[TEAM]","([TEAM])"], special_tokens=True)
        self.opt_model = AutoModelForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        #self.opt_model.resize_token_embeddings(len(self.tokenizer))
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        # tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        
        #self.llama_model.parallelize()
        
        self.ln_vision = LayerNorm(num_features)
        self.num_query_tokens = num_query_tokens,
        self.num_video_query_token = num_video_query_token
        self.inference = inference
        
        
         # Initialize video Q-former
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,
                                                                             vision_width=num_features,
                                                                             num_hidden_layers =6)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # llama projection
        self.opt_proj = nn.Linear(
            self.video_Qformer.config.hidden_size, 512
        )
        # video frame positional embedding
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, num_features)
        self.window = window

        # move to device
        self.opt_model = self.opt_model.to(self.device)
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.video_Qformer = self.video_Qformer.to(self.device)
        self.opt_proj = self.opt_proj.to(self.device)
        self.ln_vision = self.ln_vision.to(self.device)
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.video_frame_position_embedding = self.video_frame_position_embedding.to(self.device)
        
        self.generate_dtype = None

        
        
    
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
        
    
            
        
        video_features = batch['vid_features'].to(self.device) #B,T,P,D or [T,P,D]
        input_ids= batch['input_ids']#Bxmax(T)
        atts_opt = batch['attention_mask']  #Bxmax(T)
        targets= batch['labels'] #Bxmax(T)
        commentary = batch['commentary']
        
        self.generate_dtype = atts_opt.dtype
  
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
        
        inputs_opt = self.opt_proj(video_hidden)
        
        #inputs_embeds = self.opt_model.model.decoder.embed_tokens(input_ids)
        
        # targets = opt_tokens.input_ids.masked_fill(
        #     input_ids == self.opt_tokenizer.pad_token_id, -100
        # ).to(self.device)
        
        # empty_targets = (
        #     torch.ones(atts_opt.size(), dtype=torch.long).to(self.device).fill_(-100)
        # )
        
        # targets = torch.cat([empty_targets, targets], dim=1)
        
        
        
        
        
        if self.inference:
            return self.generate_text(inputs_opt)
        
        if validating:
            output_text = self.generate_text(inputs_opt,validation=True)
            return output_text, batch['commentary']
        
        #atts_llama
        
        visual_label = torch.full((batch_size, self.num_video_query_token), -100, dtype=torch.long)
        concat_targets = torch.cat((visual_label, targets), dim=1).to(self.device)
        
        temp_input_ids = input_ids.clone().to(self.device)
        targets_embeds = self.opt_model.model.decoder.embed_tokens(temp_input_ids)
        

        embedding_cat = torch.cat((inputs_opt, targets_embeds), dim=1)
        mask_prefix = torch.ones(batch_size, self.num_video_query_token, dtype=atts_opt.dtype)
        mask = torch.concat((mask_prefix, atts_opt), dim=1).to(self.device)
    
        
        # original_stdout = sys.stdout
        # sys.stdout = io.StringIO()
        
        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=embedding_cat,
                attention_mask=mask,
                return_dict=True,
                labels=concat_targets,
            )
            
        # sys.stdout = original_stdout
        loss = outputs.loss
    
        return loss
        

        
    def generate_text(self, inputs_opt,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        validation=False):
        
        batch_size = inputs_opt.shape[0]
    
        mask_prefix = torch.ones(batch_size, self.num_video_query_token, dtype=torch.float32).to(self.device)
        
        with self.maybe_autocast():
            outputs = self.opt_model.generate(
                    inputs_embeds=inputs_opt, 
                    attention_mask=mask_prefix,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=256,
                    min_length=min_length,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
        
        output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
        output_text = [text.strip() for text in output_text]
        
        if validation:

            return output_text
            
        
        return output_text
        
       
        
        
    
    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
        
    # def generate_text(self, inputs_opt):
    #     start_embeds = self.opt_model.model.embed_tokens(torch.tensor([128000]).to(self.device))
    #     # inputs_opt_with_s = torch.cat([inputs_opt, start_embeds.expand(inputs_opt.size(0), -1, -1)], dim=1).to(dtype=torch.bfloat16)
    #     # temp_res_tokens = self.opt_model.generate(
    #     #     renormalize_logits=True,
    #     #     inputs_embeds=inputs_opt_with_s,
    #     #     max_new_tokens=128,
    #     #     num_beams=5,
    #     #     do_sample=True,
    #     #     min_length=5,
    #     #     top_p=0.9,
    #     #     repetition_penalty=1.0,
    #     #     length_penalty=1,
    #     #     temperature=1.0,
    #     # )
    #     # res_text = process_output_tokens(self, temp_res_tokens)
    #     # return res_text
    #     return None

    

        
