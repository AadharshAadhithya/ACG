from dataset import ACGDataset
from mymodel import MyModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import torch
import numpy as np
import random
import os
from pycocoevalcap.cider.cider import Cider
from config import config 

def eval_cider(predicted_captions, gt_captions):
    cider_evaluator = Cider()
    predicted_captions_dict =dict()
    gt_captions_dict = dict()
    for i, caption in enumerate(predicted_captions):
        predicted_captions_dict[i] = [caption]
    for i, caption in enumerate(gt_captions):
        gt_captions_dict[i] = [caption]
    _, cider_scores = cider_evaluator.compute_score(predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()

def train():
    train_dataset = ACGDataset(config.data.train_path)
    val_dataset = ACGDataset(config.data.val_path)
    
    train_data_loader = DataLoader(train_dataset, batch_size=config.data.train_batch_size,
                                   num_workers=config.data.train_num_workers,
                                   drop_last=False, shuffle=True, pin_memory=True, 
                                   collate_fn=train_dataset.collator)
    val_data_loader = DataLoader(val_dataset, batch_size=config.data.val_batch_size,
                                 num_workers=config.data.val_num_workers,
                                 drop_last=True, shuffle=True, pin_memory=True,
                                 collate_fn=val_dataset.collator)
    
    model = MyModel().to('cuda')
    
    # if args.continue_train:
    #     model.load_state_dict(torch.load(args.load_ckpt))
    
    optimizer = AdamW(model.parameters(), lr=config.training.lr)
    
    # os.makedirs(args.model_output_dir, exist_ok=True)
    
    max_val_CIDEr = max(float(0), config.training.pre_max_CIDEr)
    
    for epoch in range(config.training.num_epochs):
        
        model.train()
        train_loss_accum = 0.0
        train_pbar = tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} Training')
        
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            try:
                loss = model(samples)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
                train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                avg_train_loss = train_loss_accum / len(train_data_loader)
            except:
                pass
            
        # model.eval()
        # val_CIDEr = 0.0
        # val_pbar = tqdm(val_data_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} Validation')
        # with torch.no_grad():
        #     for samples in val_pbar:
        #         temp_res_text, anonymized = model(samples, True)
        #         cur_CIDEr_score = eval_cider(temp_res_text,anonymized)
        #         val_CIDEr += sum(cur_CIDEr_score)/len(cur_CIDEr_score)
        #         val_pbar.set_postfix({"Scores": f"|C:{sum(cur_CIDEr_score)/len(cur_CIDEr_score):.4f}"})
                
        # avg_val_CIDEr = val_CIDEr / len(val_data_loader)
        # print(f"Epoch {epoch+1} Summary: Average Training Loss: {avg_train_loss:.3f}, Average Validation scores: C:{avg_val_CIDEr*100:.3f}")
            
            
        # if epoch % 5 == 0:
        #     file_path = f"{args.model_output_dir}/model_save_{epoch+1}.pth"
        #     save_matchvoice_model(model, file_path)

        # if avg_val_CIDEr > max_val_CIDEr:
        #     max_val_CIDEr = avg_val_CIDEr
        #     file_path = f"{args.model_output_dir}/model_save_best_val_CIDEr.pth"
        #     save_matchvoice_model(model, file_path)
    
    
    
    
    
train()