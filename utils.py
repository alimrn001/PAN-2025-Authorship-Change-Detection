import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm 
import os
import ast

model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

DATASET_LOC_PAIR = '../dataset/v2/Pairwise_no_oversampling/'
OUTPUT_LOC = '../dataset/v2/embeddings_no_oversampling/'


def extract_filename_and_category_from_input_filename(filename):
    category = os.path.normpath(filename).split(os.sep)[0]  
    filename_no_ext = os.path.splitext(os.path.basename(filename))[0]  
    return category, filename_no_ext


def generate_pairwise_data(filename, output_dir, arr_string):
    # data_list = ast.literal_eval(arr_string)
    os.makedirs(output_dir, exist_ok=True)

    json_array = [{"s1": arr_string[i], "s2": arr_string[i + 1]} for i in range(len(arr_string) - 1)]
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(json_array, f, indent=4, ensure_ascii=False)


def get_labse_embedding(sentence):
    inputs = tokenizer(
        sentence, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(device)  
    
    with torch.no_grad():
        output = model(**inputs)
    
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings.squeeze().cpu().numpy()


def generate_embeddings_dataset_file(input_loc, output_loc, has_ground_truth):
    with open(input_loc, 'r', encoding='utf-8-sig') as f:
        dataset = json.load(f)

    data_with_embeddings = []

    for idx, item in tqdm(enumerate(dataset), desc="Generating Embeddings", unit="item"):
        s1 = item['s1']
        s2 = item['s2']

        s1_embed = get_labse_embedding(s1)
        s2_embed = get_labse_embedding(s2)

        if has_ground_truth:
            data_with_embeddings.append({
                "idx": idx,
                "label": item["label"],
                "s1": s1,
                "s2": s2,
                "s1_embedding": s1_embed,
                "s2_embedding": s2_embed
            })
        else:
            data_with_embeddings.append({
                "idx": idx,
                "s1": s1,
                "s2": s2,
                "s1_embedding": s1_embed,
                "s2_embedding": s2_embed
            })

    os.makedirs(os.path.dirname(output_loc), exist_ok=True)
    np.save(output_loc, data_with_embeddings, allow_pickle=True)


def create_LaBSE_embeddings(has_ground_truth, pairwise_dataset_dir, embeddings_dataset_dir, filename):
    os.makedirs(pairwise_dataset_dir, exist_ok=True)           
    input_loc = os.path.join(pairwise_dataset_dir, f"{filename}.json")
    output_loc = os.path.join(embeddings_dataset_dir, f"{filename}.npy")
    
    print(input_loc)
    # input_loc = f"{filename}.json"
    # output_loc = f"{OUTPUT_LOC}{type[:-1]}_{train_val[:-1]}.npy"
    generate_embeddings_dataset_file(input_loc, output_loc, has_ground_truth)
           