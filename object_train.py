import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import ElectraModel, ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
import json
import glob
from tqdm import tqdm
from datetime import datetime
import time

def load_json_data(directory):
    json_files = glob.glob(directory + "*.json")
    data = []
    for json_file in tqdm(json_files, desc=f"Reading JSON files from {directory}"):
        with open(json_file, "r", encoding="utf-8") as file:
            data_item = json.load(file)
            data.append(data_item)
    return data

# 각 데이터 세트를 로드
train_data_directory = "/data/train/train/"
test_data_directory = "/data/train/test/"
validation_data_directory = "/data/train/val/"

train_data = load_json_data(train_data_directory)
test_data = load_json_data(test_data_directory)
val_data = load_json_data(validation_data_directory)

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
            ner_tags = ['NO'] * len(tokens)

            # 어노테이션을 통해 NER 태그 설정
            for annotation in annotations:
                start_pos = annotation["startPos"]
                end_pos = annotation["endPos"]
                tag_class = annotation["Tagclass"]

                if tag_class != 'O':
                    continue

                token_start_pos = len(tokenizer.tokenize(sentence[:int(start_pos)]))
                token_end_pos = len(tokenizer.tokenize(sentence[:int(end_pos) + 1])) - 1

                if token_start_pos < len(tokens) and token_end_pos < len(tokens):
                    ner_tags[token_start_pos] = f"B-{tag_class}"
                    for i in range(token_start_pos + 1, token_end_pos + 1):
                        ner_tags[i] = f"I-{tag_class}"

            processed_data.append((sentence, tokens, ner_tags))
    return processed_data

train_processed = process_data(train_data)
test_processed = process_data(test_data)
val_processed = process_data(val_data)

#1 NERDataset 클래스 정의
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = ['NO', 'B-O', 'I-O']
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
    
train_dataset = NERDataset(train_processed, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

val_dataset = NERDataset(val_processed, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

test_dataset = NERDataset(test_processed, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = 1e-5
EPOCHS = 3
# 모델 초기화
model = KoBERTBiGRU(tokenizer=tokenizer, model=model, target_vocab_size=3)
model = model.to(device)
# 학습 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
epochs = EPOCHS

def group_emotion_labels(labels, label_map):
    return [1 if label_map.get(label, 'NO') in ['B-O', 'I-O'] else 0 for label in labels]
#돌아가는거
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
prev_val_loss = float('inf')
increasing_count = 0
early_stop_limit = 3

all_preds = []  
all_labels = [] 
print("모델 시작 시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print(f"Num Epochs = {EPOCHS}\n")
print(f"Learning rate = {LEARNING_RATE}\n")
print(f"Instantaneous batch size per device = {train_loader.batch_size}\n")

try:
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc='Training', position=0, leave=True)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'].mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training loss: {average_train_loss}")

        torch.save(model, f'/pth/Object{epoch+1}.pth')

        model.eval()
        all_preds.clear() 
        all_labels.clear()
        val_loss = 0
        progress_bar = tqdm(val_loader, desc='Validation', position=0, leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids, attention_mask, labels = [item.to(device) for item in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs['loss'].mean()
                preds = outputs['logits'].argmax(dim=-1)

        average_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {average_val_loss}")

        if average_val_loss >= prev_val_loss:
            increasing_count += 1
        else:
            increasing_count = 0

        prev_val_loss = average_val_loss

        if increasing_count >= early_stop_limit:
            print("Early stopping triggered")
            break

finally:
    model.eval()  # 모델을 평가 모드로 설정
    true_labels = [label for label in all_labels if label != -100]
    pred_labels = [pred for pred, label in zip(all_preds, all_labels) if label != -100]
    f1 = f1_score(true_labels, pred_labels, average='macro')

    # 라벨 그룹화 및 분류 보고서 출력
    label_map = {0: 'NO', 1: 'B-O', 2: 'I-O'}  # 라벨 맵 정의

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs['logits'].argmax(dim=-1)
            all_preds.extend(preds.flatten().cpu().tolist())
            all_labels.extend(labels.flatten().cpu().tolist())

    filtered_true_labels = [label for label in all_labels if label != -100]
    filtered_pred_labels = [pred for pred, label in zip(all_preds, all_labels) if label != -100]
    grouped_true_labels = group_emotion_labels(filtered_true_labels, label_map)
    grouped_pred_labels = group_emotion_labels(filtered_pred_labels, label_map)

    # 변경된 라벨에 대한 F1 스코어 및 classification_report 계산
    report2 = classification_report(grouped_true_labels, grouped_pred_labels, target_names=['Non-Object', 'Object'])
    report = classification_report(grouped_true_labels, grouped_pred_labels, target_names=['Non-Object', 'Object'],output_dict=True)
    print(report2)
    Object_f1_score = report['Object']['f1-score']
    print(f"Object 클래스 F1 스코어: {Object_f1_score:.4f}")
    print("모델 종료 시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))