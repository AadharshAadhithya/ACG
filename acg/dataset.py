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

IGNORE_INDEX=-100

class ACGDataset(Dataset):
    
    def __init__(self,root_dir, fps = 1,
                 tokenizer_name = config.model.language_model.tokenizer_name, max_token_length=128):
        
        self.root_path = root_dir  
        self.vid_embs_dir = os.path.join(self.root_path, "vid_embs")
        self.commentary_dir = os.path.join(self.root_path, "transcriptions")
        self.tokenizer_name = tokenizer_name
        
        # the video embeddings are sampled at two embedding per second. 
        #need to hcange the below parameter as req
        self.fps = fps
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,use_auth_token=True)
        self.tokenizer.pad_token_id = 128001
        self.tokenizer.add_tokens(["[BATSMAN]","[BOWLER]", "[FIELDER]", 
                                   "[UMPIRE]", "[VENUE]" ], special_tokens=True)
        self.max_token_length = max_token_length
        
        self.vid_ids = self._get_valid_vidids()
        
        
    def __getitem__(self,idx):
        
        vid_id = self.vid_ids[idx]
        
        #load npy file
        features_path = os.path.join(self.vid_embs_dir, vid_id+"_embeddings.npy")
        features = np.load(features_path) # Time, Patches(256+cls), Dimension
        features = self._resample_features(features,self.fps)
        
        
        #load commentary for hte vid file
        commentary = self._get_vid_commentary(vid_id)
        
        tokens = self.tokenizer(commentary,return_tensors="pt", 
                                max_length=self.max_token_length,truncation=True).input_ids[0]
        
    
        return {'vid_features': features, "tokens": tokens, "commentary": commentary}
  
        
    def __len__(self):
        return len(self.vid_ids)
    
    def _resample_features(self, features, fps, time_step=0.5):
        # Calculate the new time interval between samples
        new_interval = time_step / fps  # How often to sample (in seconds)
        
        # Calculate the index step (how many original time steps to skip)
        step = int(1 / new_interval)
        
        # Create an array of indices for resampling based on fps
        new_time_indices = np.arange(0, features.shape[0], step)
        
        # Return the features sampled at the new time indices
        return features[new_time_indices]
    
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
        
        file_path = os.path.join(self.commentary_dir, f"{vid_id}.txt")
        
        
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
        
        
        
        # Get the list of files in both directories
        vid_files = os.listdir(self.vid_embs_dir)
        com_files = os.listdir(self.commentary_dir)
        
        # Get the base filenames without extensions for .npy files
        emb_files = {os.path.splitext(f)[0].split('_embeddings')[0] for f in vid_files if f.endswith('_embeddings.npy')}
        
        # Get the base filenames without extensions for .txt files
        com_files = {os.path.splitext(f)[0] for f in com_files if f.endswith('.txt')}
        
      
        # Find the intersection of the two sets
        valid_vidids = emb_files.intersection(com_files)
        
        
        return list(valid_vidids)
    
    
    def collator(self,batch):
        
        out_batch= {}

        input_ids = [
            torch.cat((torch.tensor([self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]),
                        instance["tokens"],
                        torch.tensor([self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")]))) for instance in batch] 

        labels = copy.deepcopy(input_ids)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))

        attention_mask=input_ids.ne(self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
        
        vid_features = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(instance['vid_features']) for instance in batch], batch_first=True )
        
        commentaries = [instance['commentary'] for instance in batch]
        
       


        # if 'vid_features' in batch[0]:
        #     features = [torch.from_numpy(instance['vid_features']) for instance in batch]
        # if all(x is not None and x.shape == features[0].shape for x in features):
        #     out_batch['vid_features'] = torch.stack(features)
        # else:
        #     out_batch['vid_features'] = features
        
        
            
        out_batch['vid_features'] = vid_features
        out_batch['input_ids']=input_ids
        out_batch['attention_mask'] =attention_mask
        out_batch['labels']=labels
        out_batch['commentary'] = commentaries
        return out_batch
     