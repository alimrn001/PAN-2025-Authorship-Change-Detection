#!/usr/bin/env python3
#!/usr/bin/env python3
import json
import torch
import joblib
import numpy as np
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import AutoModel, AutoTokenizer
from gensim.models.fasttext import load_facebook_model
import click
import re
import pandas as pd
from utils import *
from huggingface_hub import hf_hub_download


hf_repo_id = "alimrn001/pan2025-multi-author-models"
fasttext_path = './FastText/crawl-300d-2M-subword.bin'
pairwise_dataset_path = './dataset/pairwise/'
labse_embeddings_path = './dataset/embeddings/'
models_parameter_path = './params/' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SiameseLSTM(torch.nn.Module):
    def __init__(self, embedding_dim=300, hidden_size=128, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2)
        )

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[0]
        h_backward = h_n[1]
        return torch.cat((h_forward, h_backward), dim=1)

    def forward(self, s1, s2):
        s1_encoded = self.encode(s1)
        s2_encoded = self.encode(s2)
        diff = torch.abs(s1_encoded - s2_encoded)
        return self.fc(diff)


class CrossAttnLaBSE(torch.nn.Module):
    def __init__(self, model_name="sentence-transformers/LaBSE", dropout_rate=0.1, num_heads=8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden * 5, 2)

    def forward(self, s1_list, s2_list):
        enc1 = self.tokenizer(s1_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.classifier.weight.device)
        enc2 = self.tokenizer(s2_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.classifier.weight.device)
        out1 = self.bert(**enc1).last_hidden_state
        out2 = self.bert(**enc2).last_hidden_state
        cls1 = out1[:, 0, :]
        cls2 = out2[:, 0, :]
        cross12 = self.cross_attn(query=cls1.unsqueeze(1), key=out2, value=out2)[0].squeeze(1)
        cross21 = self.cross_attn(query=cls2.unsqueeze(1), key=out1, value=out1)[0].squeeze(1)
        diff = torch.abs(cls1 - cls2)
        fused = torch.cat([cls1, cls2, diff, cross12, cross21], dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits


def load_fasttext(path):
    fasttext_path = hf_hub_download(repo_id=hf_repo_id, filename="FastText/crawl-300d-2M-subword.bin")
    # Load model
    return load_facebook_model(fasttext_path).wv


def sentence_to_vector(sentence, fasttext_model, max_len=100, embedding_dim=300):
    tokens = re.findall(r"\w+|[^\w\s]", sentence.lower())
    vecs = [fasttext_model[token] if token in fasttext_model else np.zeros(embedding_dim) for token in tokens[:max_len]]
    vecs += [np.zeros(embedding_dim)] * (max_len - len(vecs))
    return np.array(vecs)


def load_models(subtask, model_dir, device):
    # labse_path = Path(model_dir) / f"{subtask}/labse_{subtask}.pth"
    # siamese_path = Path(model_dir) / f"{subtask}/siamese_{subtask}.pth"
    # ensemble_path = Path(model_dir) / f"{subtask}/ensemble_{subtask}.pkl"
    
    labse_path = hf_hub_download(repo_id=hf_repo_id, filename=f"{subtask}/labse_{subtask}.pth")
    siamese_path = hf_hub_download(repo_id=hf_repo_id, filename=f"{subtask}/siamese_{subtask}.pth")
    ensemble_path = hf_hub_download(repo_id=hf_repo_id, filename=f"{subtask}/ensemble_{subtask}.pkl")

    siamese = SiameseLSTM().to(device)
    siamese.load_state_dict(torch.load(siamese_path, map_location=device))
    siamese.eval()

    labse = CrossAttnLaBSE().to(device)
    labse.load_state_dict(torch.load(labse_path, map_location=device))
    labse.eval()

    ensemble = joblib.load(ensemble_path)

    return siamese, labse, ensemble


def predict_change_from_embeddings(
    index,  # index of the pair in .npy file
    s1, s2,
    siamese_model,
    labse_model,
    ensemble_model,
    fasttext_model,
    labse_embeddings,
    device
):
    # Siamese model uses FastText
    s1_vec = sentence_to_vector(s1, fasttext_model)
    s2_vec = sentence_to_vector(s2, fasttext_model)
    s1_tensor = torch.tensor(s1_vec, dtype=torch.float32).unsqueeze(0).to(device)
    s2_tensor = torch.tensor(s2_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        siamese_probs = torch.softmax(siamese_model(s1_tensor, s2_tensor), dim=1).cpu().numpy()[0]

    # LaBSE uses pre-computed embeddings from .npy
    s1 = labse_embeddings[index]['s1']
    s2 = labse_embeddings[index]['s2']

    with torch.no_grad():
       labse_logits = labse_model([s1], [s2])
       labse_probs = torch.softmax(labse_logits, dim=1).cpu().numpy()[0]
    
    # Ensemble
    features = np.concatenate([siamese_probs, labse_probs])
    prediction = ensemble_model.predict([features])[0]

    return int(prediction)


def run_baseline(problems: "pd.DataFrame", output_path: Path, pred: int, fasttext_model):
    """
    Write predictions to solution files in the format:
    {
    "changes": [0, 0, 0, 1]
    }

    :param problems: dictionary of problem files with ids as keys
    :param output_path: output folder to write solution files
    :param pred: the static prediction
    """
    print(f'Write outputs {len(problems)} problems to to {output_path}.')
    for _, i in problems.iterrows():
        
        set_type, filename_no_ext = extract_filename_and_category_from_input_filename(i['file'])
    
        sentences = i['problem'].splitlines()
        sentences = [s.rstrip('\n') for s in sentences] # removes blank senteces
        
        generate_pairwise_data(f'{filename_no_ext}.json', f"{pairwise_dataset_path}{set_type}/", sentences)
        
        create_LaBSE_embeddings(False, f"{pairwise_dataset_path}{set_type}/", f"{labse_embeddings_path}{set_type}/", filename_no_ext)
        
        changes = [] #prediction list
        
        siamese_model, labse_model, ensemble_model = load_models(set_type, models_parameter_path, device)
        
        # fasttext_model = load_fasttext(fasttext_path)
        
        labse_embedding_path = f"{labse_embeddings_path}{set_type}/{filename_no_ext}.npy"
        labse_embeddings = np.load(labse_embedding_path, allow_pickle=True)
        
        for idx in range(len(sentences) - 1):
            s1 = sentences[idx]
            s2 = sentences[idx + 1]
            label = predict_change_from_embeddings(
                idx, s1, s2,
                siamese_model,
                labse_model,
                ensemble_model,
                fasttext_model,
                labse_embeddings,
                device
            )
            changes.append(label)
            
        output_file = output_path / i["file"].replace("/problem-", "/solution-problem-").replace(".txt", ".json").replace("/train/", "/").replace("/test/", "/").replace("/validation/", "/")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as out:
            paragraphs = i["paragraphs"]
            prediction = {'changes': changes}
            #prediction = {'changes': [pred for _ in range(len(paragraphs) -1)]}
            out.write(json.dumps(prediction))


@click.command()
@click.option('--dataset', default='multi-author-writing-style-analysis-2025/multi-author-writing-spot-check-20250503-training', help='The dataset to run predictions on (can point to a local directory).')
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent))), help='The file where predictions should be written to.')
@click.option('--predict', default=0,  help='The prediction to make.')
def main(dataset, output, predict):
    
    fasttext_model = load_fasttext(fasttext_path)
    tira = Client()
    
    # alternatively, you can still load the data programmatically, i.e., without tira.pd.inputs.
    # See https://github.com/pan-webis-de/pan-code/blob/master/clef24/multi-author-analysis/baselines/baseline-static-prediction.py
    input_df = tira.pd.inputs(dataset, formats=["multi-author-writing-style-analysis-problems"])

    for subtask in ["easy", "medium", "hard"]:
        print(f"Generating predictions for: {subtask}")
        run_baseline(input_df[input_df["task"] == subtask], Path(output), predict, fasttext_model)

if __name__ == '__main__':
    main()

