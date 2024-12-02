import os
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm

# Function to extract assistant's response
def extract_assistant_response(text):
    """
    Extracts the assistant's response using regex.
    The response is located between <|start_header_id\|>assistant<|end_header_id\|> and <|eot_id\|>.
    """
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

# Model and tokenizer setup
model_id = "meta-llama/Llama-3.1-8B-Instruct"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prompt for the model
prompt = """
You will be provided with cricket commentary. Your task is to standardize the commentary by rephrasing it into a formal and concise tone. Remove personal identifiers and replace them with the following special tokens: [BATSMAN], [BOWLER], [FIELDER], [UMPIRE], [VENUE]. Ensure that the commentary remains concise and retains its original meaning.

If no commentary or irrelevant text is provided, output "None".
"""

# Function to load and process JSON files
def load_commentaries_from_json(folder_path):
    """
    Load and combine the `text` fields from all JSON files in a folder.
    Returns a list of (filename, commentary) pairs.
    """
    all_commentaries = []
    filenames=[]
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            filenames.append(file_name.split(".")[0])
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                # Combine all `text` fields to form one commentary
                commentary = " ".join(item["text"].strip() for item in data if "text" in item)
                all_commentaries.append((file_name, commentary))
    return all_commentaries,filenames

# Function to perform batch inference
import os

# def batch_inference(commentaries, filenames, dest_dir, batch_size=4):
#     """
#     Perform inference in batches and save responses to text files.
#     """
#     results = []
#     for i in tqdm(range(0, len(commentaries), batch_size)):
#         batch = commentaries[i : i + batch_size]
#         batch_filenames = filenames[i:i + batch_size]
        
#         # Prepare batch messages
#         messages_batch = [
#             [{"role": "system", "content": prompt}, {"role": "user", "content": f"Commentary: {c[1]}"}]
#             for c in batch
#         ]

#         # Prepare inputs for the model
#         text_batch = tokenizer.apply_chat_template(messages_batch, tokenize=False)
#         inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True)
#         inputs_batch = {key: tensor.to(device) for key, tensor in inputs_batch.items()}

#         # Generate responses
#         generated_ids_batch = model_nf4.generate(
#             **inputs_batch,
#             max_new_tokens=150,
#             do_sample=True,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id
#         )
        
#         # Decode and extract responses
#         outputs = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=False)
#         parsed_outs = [extract_assistant_response(output) for output in outputs]

#         # Save each response to the corresponding file in dest_dir
#         for j, parsed_out in enumerate(parsed_outs):
#             filename = batch_filenames[j]
#             file_path = os.path.join(dest_dir, f"{filename}.txt")
#             with open(file_path, "w") as f:
#                 f.write(parsed_out)

#         # Add filename and response pair to results
#         results.extend(zip([c[0] for c in batch], parsed_outs))
    
#     return results

def batch_inference(commentaries, filenames, dest_dir, batch_size=4):
    """
    Perform inference in batches and save responses to text files. Skip the entire batch if all files already exist.
    """
    results = []
    for i in tqdm(range(0, len(commentaries), batch_size)):
        batch = commentaries[i : i + batch_size]
        batch_filenames = filenames[i:i + batch_size]
        
        # Check if the output files already exist for the batch, and skip the entire batch if so
        existing_files = [
            os.path.join(dest_dir, f"{filename}.txt") for filename in batch_filenames if os.path.exists(os.path.join(dest_dir, f"{filename}.txt"))
        ]
        
        # If all files in the batch exist, skip the entire batch
        if len(existing_files) == len(batch_filenames):
            print(f"Skipping batch {i // batch_size + 1} as all files already exist: {', '.join(existing_files)}")
            continue
        
        # Prepare batch messages
        messages_batch = [
            [{"role": "system", "content": prompt}, {"role": "user", "content": f"Commentary: {c[1]}"}]
            for c in batch
        ]

        # Prepare inputs for the model
        text_batch = tokenizer.apply_chat_template(messages_batch, tokenize=False)
        inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True)
        inputs_batch = {key: tensor.to(device) for key, tensor in inputs_batch.items()}

        # Generate responses
        generated_ids_batch = model_nf4.generate(
            **inputs_batch,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and extract responses
        outputs = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=False)
        parsed_outs = [extract_assistant_response(output) for output in outputs]

        # Save each response to the corresponding file in dest_dir
        for j, parsed_out in enumerate(parsed_outs):
            filename = batch_filenames[j]
            file_path = os.path.join(dest_dir, f"{filename}.txt")

            # Skip saving if the file already exists
            if os.path.exists(file_path):
                continue  # Already skipped above, but included for safety

            # Save the response to the file
            with open(file_path, "w") as f:
                f.write(parsed_out)

        # Add filename and response pair to results
        results.extend(zip([c[0] for c in batch], parsed_outs))
    
    return results



# Main function
def process_directory_and_run_inference(folder_path, destination_dir, batch_size=4):
    """
    Process a directory of JSON files, run inference, and save the results.
    """
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Load all commentaries with filenames
    commentaries,filenames = load_commentaries_from_json(folder_path)

    # Run batch inference
    results = batch_inference(commentaries,filenames, destination_dir, batch_size=batch_size)

    # # Save results to destination directory
    # for file_name, response in results:
    #     output_file_path = os.path.join(destination_dir, file_name.replace(".json", ".txt"))
    #     with open(output_file_path, "w") as f:
    #         f.write(response)
    #     print(f"Processed {file_name} -> Saved to {output_file_path}")

# Example usage
folder_path = "/home/ACG/data/raw/whisperx_transcripts"  # Replace with your JSON folder path
destination_dir = "/home/ACG/data/pre_2/standardized_transcripts"  # Replace with your destination folder path
process_directory_and_run_inference(folder_path, destination_dir, batch_size=2)
