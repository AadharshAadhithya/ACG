import os
import random
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from transformers import AutoTokenizer
import copy
import os
from config import config

class ACGDataset(Dataset):
    
    def __init__(self,max_token_length=128):
        
        self.root_path = config.data.train_path 
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.language_model.tokenizer_name,use_auth_token=True)
        self.tokenizer.pad_token_id = 128001
        self.tokenizer.add_tokens(["[PLAYER]","[TEAM]"], special_tokens=True)
        self.max_token_length = max_token_length
        
        self.vid_ids = self._get_valid_vidids()
        
        
    def __getitem__(self,idx):
        
        vid_id = self.vid_ids[idx]
        
        #load npy file
        features_path = os.path.join(self.root_path, vid_id+"_embeddings.npy")
        features = np.load(features_path) # Time, Patches(256+cls), Dimension
        
        #load commentary for hte vid file
        commenatry = self._get_vid_commentary(vid_id)
        
        tokens = self.tokenizer(commenatry,return_tensors="pt", max_length=self.max_token_length,truncation=True).input_ids[0]
        
    
        return {'vid_features': features, "tokens": tokens, "commentary": commenatry}
  
        
    def __len__(self):
        return len(self.vid_ids)
    
    def _get_vid_commentary(self, vid_id):
        """
        Reads and returns the text from the commentary file associated with the given vid_id.

        Args:
            vid_id (str): The video ID whose commentary is to be fetched.

        Returns:
            str: The content of the commentary file.

        Raises:
            FileNotFoundError: If the .txt file for the given vid_id does not exist.
        """
        
        file_path = os.path.join(self.root_path, f"{vid_id}.txt")
        
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Commentary file not found for vid_id: {vid_id}")

        
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    


    def _get_valid_vidids(self):
        """
        Finds video IDs (vidids) in the given directory for which both .txt and _embeddings.npy files are present.

        Args:
            directory (str): Path to the directory containing the files.

        Returns:
            list: List of vidids with both .txt and _embeddings.npy files present.
        """
        
        files = os.listdir(self.root_path)
        
        
        txt_files = {os.path.splitext(f)[0] for f in files if f.endswith('.txt')}
        npy_files = {os.path.splitext(f)[0].rsplit('_embeddings', 1)[0] for f in files if f.endswith('_embeddings.npy')}
        
        
        valid_vidids = txt_files.intersection(npy_files)
        
        return list(valid_vidids)
    
    
    def collator(self,batch):
        
     out_batch= {}
     
     labels = copy.deepcopy(input_ids)
     
     labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        

    
     input_ids = [
            torch.cat((torch.tensor([self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]),
                       instance["tokens"],
                       torch.tensor([self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")]))) for instance in batch] 
     
     input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
     
     attention_mask=input_ids.ne(self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
     
     
      if 'vid_features' in batch[0]:
            features = [instance['vid_features'] for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                out_batch['vid_features'] = torch.stack(features)
            else:
                out_batch['vid_features'] = features
                
       out_batch['input_ids']=input_ids
       out_batch['attention_mask'] =attention_mask
       out_batch['labels']=labels
        return out_batch
     
   



    
