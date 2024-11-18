import torch
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import ElectraModel, ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
import pickle
import json
import glob
from tqdm import tqdm
from datetime import datetime

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

import torch.nn as nn

class KoBERTBiGRU(nn.Module):
    def __init__(self, tokenizer, model, hidden_dim=768, target_vocab_size=3, gru_layers=3, dropout=0.3):
        super(KoBERTBiGRU, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = model
        
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size, 
                          hidden_size=hidden_dim, 
                          num_layers=gru_layers, 
                          bidirectional=True, 
                          batch_first=True, 
                          dropout=dropout if gru_layers > 1 else 0)
        self.batchnorm = nn.BatchNorm1d(hidden_dim * 2)
        self.hidden2tag = nn.Linear(hidden_dim * 2, target_vocab_size)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)  # -100 is typically used as a label to ignore in NLP tasks

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        gru_output, _ = self.gru(bert_output.last_hidden_state)
        gru_output = self.batchnorm(gru_output.permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.hidden2tag(gru_output)
        
        if labels is not None:
            # Compute the loss. We reshape logits and labels to [-1, num_labels] before passing them to the loss function
            # This is to align the dimensions as CrossEntropy expects inputs in this shape.
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_function(active_logits, active_labels)
            return {"loss": loss, "logits": logits}
        else:
            # For predictions, you might want to compute the argmax over the logits
            predictions = logits.argmax(dim=-1)
            return {"logits": logits, "predictions": predictions}

def load_json_data(directory):
    json_files = glob.glob(directory)
    json_files.sort()
    data = []
    for json_file in tqdm(json_files, desc=f"Reading JSON files from {directory}"):
        with open(json_file, "r", encoding="utf-8") as file:
            data_item = json.load(file)
            data.append(data_item)
    return data

# 각 데이터 세트를 로드
test_data_directory = "/data/test_data/*.json"

test_data = load_json_data(test_data_directory)

def process_data(data_list):
    processed_data = []
    for data_item in tqdm(data_list):
        # 각 문장에 대해
        for sentence_info in data_item["docu_info"]["sentences"]:
            sentence = sentence_info["sentence"]

            # 문장이 None인지 확인
            if sentence is None:
                continue

            annotations = sentence_info["annotations"]
            if annotations is None:
                continue

            # 토크나이징
            tokens = tokenizer.tokenize(sentence)

            # NER 태그 초기화
            ner_tags = ['O'] * len(tokens)

            # 어노테이션을 통해 NER 태그 설정
            for annotation in annotations:
                start_pos = annotation["startPos"]
                end_pos = annotation["endPos"]
                tag_class = annotation["Tagclass"]

                if tag_class != 'E':
                    continue

                token_start_pos = len(tokenizer.tokenize(sentence[:int(start_pos)]))
                token_end_pos = len(tokenizer.tokenize(sentence[:int(end_pos) + 1])) - 1

                if token_start_pos < len(tokens) and token_end_pos < len(tokens):
                    ner_tags[token_start_pos] = f"B-{tag_class}"
                    for i in range(token_start_pos + 1, token_end_pos + 1):
                        ner_tags[i] = f"I-{tag_class}"

            processed_data.append((sentence, tokens, ner_tags))
    return processed_data


test_processed = process_data(test_data)

#1 NERDataset 클래스 정의
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = ['O', 'B-E', 'I-E']
        self.label_map = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tokens, ner_tags = self.data[idx]
        
        labels = [self.label_map[tag] for tag in ner_tags]
        labels = labels[:self.max_length - 2]
        labels = [-100] + labels + [-100]

        inputs = self.tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        labels = labels + [-100] * (len(input_ids) - len(labels))
        labels = torch.tensor(labels, dtype=torch.long)
        
        return input_ids, attention_mask, labels
    
test_dataset = NERDataset(test_processed, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '/pth/Emotion3.pth' 
model = torch.load(model_path)
model.eval()

def group_emotion_labels(labels, label_map):
    return [1 if label_map.get(label, 'O') in ['B-E', 'I-E'] else 0 for label in labels]

all_preds = []  
all_labels = [] 

# 로그 파일 생성
log_file = '/log/emotion_model_output_log.txt'
model_start_time = datetime.now()
print("모델 시작 시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
with torch.no_grad(), open(log_file, 'w', encoding='utf-8') as log:
    log.write(f"모델 시작 시간: {model_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    total_batches = len(test_loader)
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs['logits'].argmax(dim=-1)
        # 진행률을 로그 파일에 기록
        progress = (batch_idx + 1) / total_batches * 100
        log.write(f"Evaluating: {progress:.2f}% | {batch_idx + 1}/{total_batches} batches processed\n")
        for i in range(input_ids.size(0)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            predicted_tags = preds[i].cpu().numpy()
            true_tags = labels[i].cpu().numpy()

            # 특수 토큰을 제외한 문장 생성
            sentence_tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
            sentence = " ".join(sentence_tokens)

            # 예측된 엔터티 그룹화
            predicted_entities = []
            entity = []
            for token, tag_idx in zip(tokens, predicted_tags):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                label = test_dataset.labels[tag_idx]
                if label == "B-E":
                    if entity:
                        predicted_entities.append(''.join(entity).replace('##', ''))
                        entity = []
                    entity.append(token)
                elif label == "I-E":
                    entity.append(token)
                elif entity:
                    predicted_entities.append(''.join(entity).replace('##', ''))
                    entity = []
            if entity:
                predicted_entities.append(''.join(entity).replace('##', ''))

            # 실제 엔터티 그룹화
            true_entities = []
            entity = []
            for token, tag in zip(tokens, true_tags):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                label = test_dataset.labels[tag] if tag != -100 else None
                if label == "B-E":
                    if entity:
                        true_entities.append(''.join(entity).replace('##', ''))
                        entity = []
                    entity.append(token)
                elif label == "I-E":
                    entity.append(token)
                elif entity:
                    true_entities.append(''.join(entity).replace('##', ''))
                    entity = []
            if entity:
                true_entities.append(''.join(entity).replace('##', ''))
            log.write(f"문장: {sentence}\n")
            log.write(f"실제 엔터티: {', '.join(true_entities)}\n")
            log.write(f"예측된 엔터티: {', '.join(predicted_entities)}\n\n")
            all_preds.extend(preds[i].flatten().cpu().tolist())
            all_labels.extend(labels[i].flatten().cpu().tolist())

    # 필터링 및 그룹화
    true_labels = [label for label in all_labels if label != -100]
    pred_labels = [pred for pred, label in zip(all_preds, all_labels) if label != -100]

    label_map = {0: 'O', 1: 'B-E', 2: 'I-E'}
    grouped_true_labels = group_emotion_labels(true_labels, label_map)
    grouped_pred_labels = group_emotion_labels(pred_labels, label_map)
    report2 = classification_report(grouped_true_labels, grouped_pred_labels, target_names=['Non-emotions', 'emotions'])
    report = classification_report(grouped_true_labels, grouped_pred_labels, target_names=['Non-emotions', 'emotions'], output_dict=True)
    Object_f1_score = report['emotions']['f1-score']
    log.write(f"성능 평가 결과:\n{classification_report(grouped_true_labels, grouped_pred_labels, target_names=['Non-emotions', 'emotions'])}\n")
    log.write(f"emotion 클래스 F1 스코어: {Object_f1_score:.4f}\n")
    log.write(f"모델 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(report2)
    print(f"emotion 클래스 F1 스코어: {Object_f1_score:.4f}")
    print("모델 종료 시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))