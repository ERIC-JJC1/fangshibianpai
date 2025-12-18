import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- 配置 ---
DATA_PATH = "dataset_train.csv" # 确保文件名和你生成的一致
MODEL_SAVE_PATH = "checkpoints/voltage_predictor.pth"
os.makedirs("checkpoints", exist_ok=True)

# 1. 定义数据集
class PowerGridDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        # 读取数据
        self.data = pd.read_csv(csv_file)
        # 过滤掉不收敛的坏数据 (可选，或者让模型学习预测0)
        self.data = self.data[self.data['converged'] == 1].reset_index(drop=True)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['prompt']
        # 目标值：预测最低电压
        label = float(row['target_voltage'])
        
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 2. 定义模型 (LLM Feature Extractor + Regressor Head)
class VoltagePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用轻量级 DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # 冻结 LLM 大部分参数，只微调最后几层 (加快训练)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 回归头 (这里未来会换成 GP 层)
        self.head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 输出电压值
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的向量作为整句话的特征
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding).squeeze()

# 3. 训练循环
def train():
    # 检查 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载 Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 准备数据
    dataset = PowerGridDataset(DATA_PATH, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 初始化模型
    model = VoltagePredictor().to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3) # 只优化 Head
    criterion = nn.MSELoss()
    
    # 开始训练
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            preds = model(input_ids, mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                preds = model(input_ids, mask)
                loss = criterion(preds, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  -> Model Saved!")

if __name__ == "__main__":
    train()