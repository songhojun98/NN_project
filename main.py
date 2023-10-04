import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from tqdm import tqdm
from dataset import train_combined_loader, test_combined_loader
from models import *
from torch.optim.lr_scheduler import StepLR
from scipy.stats import pearsonr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(dataloader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.view(-1)
        loss = criterion(output, data.affinity)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs  # 전체 배치에 대한 loss
    return total_loss / len(dataloader.dataset)  # 평균 loss

def test(model, dataloader):
    model.eval()
    total_loss = 0
    sample_predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Testing", leave=False):
            data = data.to(device)
            output = model(data)
            output = output.view(-1)
            loss = criterion(output, data.affinity)
            total_loss += loss.item() * data.num_graphs
            for true_val, pred_val in zip(data.affinity, output):
                sample_predictions.append((true_val, pred_val))
    return total_loss / len(dataloader.dataset), sample_predictions[:10]  # 평균 loss 및 단백질-리간드 쌍 5개 예측 값 확인

BEST_MODEL_PATH = "best_model"  # 모델을 저장할 폴더 경로
os.makedirs(BEST_MODEL_PATH, exist_ok=True)  # 폴더가 없으면 생성

def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    # Check if using CPU or GPU
    if device.type == 'cuda':
        print("Using GPU for model training.")
    else:
        print("Using CPU for model training.")
        
    # Model, Optimizer, Loss function 정의
    input_dim = train_combined_loader.dataset[0].x.size(1)
    hidden_dim = 512
    output_dim = 1

    model = GraphPoolingModel_layer_1(input_dim, hidden_dim, output_dim).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 50

    best_val_loss = float('inf')  # 초기값으로 무한대 설정
    steps_without_improvement = 0
    early_stopping_steps = 10  # Early stopping을 위한 스텝

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, num_epochs+1):
        print(f"Epoch: {epoch}")
        train_loss = train(model, train_combined_loader, optimizer, criterion)
        test_loss, sample_preds = test(model, test_combined_loader)
        
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            # Save more than just the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'steps_without_improvement': steps_without_improvement
            }, os.path.join(BEST_MODEL_PATH, "best_model.pth"))
            steps_without_improvement = 0  # 카운터 초기화
        else:
            steps_without_improvement += 1  # 손실이 감소하지 않으면 카운터 증가
        
        # Early stopping 조건 확인
        if steps_without_improvement >= early_stopping_steps:
            print("Early stopping triggered.")
            break
        
        # Learning rate scheduler 업데이트
        scheduler.step()

        # MAE 계산
        mae = sum(abs(true - pred) for true, pred in sample_preds) / len(sample_preds)

        # Pearson 상관계수 계산
        true_affinities = [true.cpu().numpy() for true, _ in sample_preds]
        pred_affinities = [pred.cpu().numpy() for _, pred in sample_preds]
        pearson_corr, _ = pearsonr(true_affinities, pred_affinities)


        # 기존 코드에서의 출력 부분
        print("--------------------------------")
        print(f"Train Loss         : {train_loss:.4f}")
        print(f"Test RMSE          : {test_loss**0.5:.4f}", flush=True)
        print(f"MAE                : {mae:.4f}", flush=True)
        print(f"Pearson Correlation: {pearson_corr:.4f}", flush=True)
        print("--------------------------------\n")

        print("\nSample Predictions for this epoch:")
        print("--------------------------------------------------", flush=True)
        for true, pred in sample_preds:
            print(f"True Affinity: {true.item():.4f}, Predicted Affinity: {pred.item():.4f}", flush=True)
        print("--------------------------------------------------", flush=True)
