import wandb  # Importing wandb for logging
from dataset import ACGDataset
from optmodel import OPTModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from llamamodel import LLAMAModel
from transformers import AdamW
import torch
import numpy as np
import random
import os
from pycocoevalcap.cider.cider import Cider
from config import config


def eval_cider(predicted_captions, gt_captions):
    print(predicted_captions,gt_captions)
    cider_evaluator = Cider()
    predicted_captions_dict = dict()
    gt_captions_dict = dict()
    for i, caption in enumerate(predicted_captions):
        print(i,caption)
        predicted_captions_dict[i] = [caption]
    for i, caption in enumerate(gt_captions):
        print(i,caption)
        gt_captions_dict[i] = [caption]
    _, cider_scores = cider_evaluator.compute_score(predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()


def save_opt_model(model, file_path):
    try:
        torch.save(model.state_dict(), file_path)
        print(f"Model successfully saved to {file_path}")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")

def save_llama_model(model, file_path):
    state_dict = model.cpu().state_dict()
    state_dict_without_llama = {}
    # 遍历原始模型的 state_dict，并排除 llama_model 相关的权重
    for key, value in state_dict.items():
        if "llama_model.model.layers" not in key:
            state_dict_without_llama[key] = value
    torch.save(state_dict_without_llama, file_path)
    model.to(model.device)


def train(args):
    # Initialize wandb
    wandb.init(project="ACGModelTraining", config=args.__dict__)
    wandb.config.update(args)
    
    train_dataset = ACGDataset(root_dir=args.train_feature_root, fps=args.fps, 
                               tokenizer_name=args.tokenizer_name, 
                               max_token_length=args.max_token_length)
    val_dataset = ACGDataset(root_dir=args.val_feature_root, fps=args.fps, 
                             tokenizer_name=args.tokenizer_name,
                             max_token_length=args.max_token_length)

    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                   num_workers=args.train_num_workers, 
                                   drop_last=False, shuffle=True, pin_memory=False, 
                                   collate_fn=train_dataset.collator)
    val_data_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, 
                                 num_workers=args.val_num_workers, drop_last=True,
                                 shuffle=True, pin_memory=False, 
                                 collate_fn=train_dataset.collator)
    
    if 'opt' in args.model_id :

        model = OPTModel(args.model_id, args.tokenizer_name,
                        num_query_tokens=args.num_query_tokens, 
                        num_video_query_token=args.num_video_query_token,
                        num_features=args.num_features, device=args.device).to(args.device)
        save_fn = save_opt_model
        
    if 'llama' in args.model_id:
        model = LLAMAModel(args.model_id, args.tokenizer_name,
                        num_query_tokens=args.num_query_tokens, 
                        num_video_query_token=args.num_video_query_token,
                        num_features=args.num_features, device=args.device, qbit_4= args.qbit_4).to(args.device)
        save_fn = save_llama_model
        

    optimizer = AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.model_output_dir, exist_ok=True)

    max_val_CIDEr = max(float(0), config.training.pre_max_CIDEr)
    
    
    # val_pbar = tqdm(val_data_loader)
    # with torch.no_grad():
    #     for samples in val_pbar:
            
    #         output_text, anonymized = model(samples, True)
    #         cur_CIDEr_score = eval_cider(output_text, anonymized)
            

    for epoch in range(args.pre_epoch, args.num_epoch):
        model.train()
        train_loss_accum = 0.0
        train_pbar = tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} Training')
        train_limit = 0
        for batch in train_pbar:
            optimizer.zero_grad()
            try:
              
                loss = model(batch)
             
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
                train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            except Exception as e:
                # print('some error')
                print(e)
                
            # train_limit += 1 
            
            # if train_limit ==4:
            #     break
        
        avg_train_loss = train_loss_accum / len(train_data_loader)
        
        model.eval()
        val_CIDEr = 0.0
        val_pbar = tqdm(val_data_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} Validation')
        with torch.no_grad():
            for samples in val_pbar:
                
                output_text, anonymized = model(samples, True)
                print( output_text, anonymized )
                cur_CIDEr_score = eval_cider(output_text, anonymized)
                val_CIDEr += sum(cur_CIDEr_score) / len(cur_CIDEr_score)
                val_pbar.set_postfix({"Scores": f"|C:{sum(cur_CIDEr_score)/len(cur_CIDEr_score):.4f}"})
                print(output_text)
                print(anonymized)
        
        avg_val_CIDEr = val_CIDEr / len(val_data_loader)
        print(f"Epoch {epoch+1} Summary: Average Training Loss: {avg_train_loss:.3f}, Average Validation CIDEr: {avg_val_CIDEr:.3f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_CIDEr": avg_val_CIDEr
        })

        if epoch % 5 == 0:
            file_path = f"{args.model_output_dir}/model_save_{epoch+1}.pth"
            save_fn(model, file_path)

        if avg_val_CIDEr > max_val_CIDEr:
            max_val_CIDEr = avg_val_CIDEr
            file_path = f"{args.model_output_dir}/model_save_best_val_CIDEr.pth"
            save_fn(model, file_path)



if __name__ == "__main__":
    import argparse
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--train_feature_root", type=str, default="../data/pre_3/train")
    parser.add_argument("--val_feature_root", type=str, default="../data/pre_3/val")
    parser.add_argument("--window", type=float, default=15)
    
   
    
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    
    
    # parser.add_argument("--tokenizer_name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model_id", type=str, default="facebook/opt-350m")
    parser.add_argument("--max_token_length", type=int, default=128)
    parser.add_argument("--train_ann_root", type=str, default="./dataset/MatchTime/train")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--train_num_workers", type=int, default=4)

    
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--val_num_workers", type=int, default=4)
   

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--num_video_query_token", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--model_output_dir", type=str, default="../ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--qbit_4", type=bool, default=True)
    
    

    # If continue training from any epoch
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--pre_max_CIDEr", type=float, default=0.0)
    parser.add_argument("--pre_epoch", type=int, default=0)
    parser.add_argument("--load_ckpt", type=str, default="./ckpt/model_save_best_val_CIDEr.pth")


    args = parser.parse_args()
    train(args)