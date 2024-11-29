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
                 num_video_query_token=32, num_features=512, 
                 device = "cuda", inference=False):
        
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
    
    def forward(self):
        pass
    
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


model  = MyModel(config.data.train_path)
        