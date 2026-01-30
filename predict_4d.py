import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
from tqdm import tqdm
import time

# Configuration
DATA_FILE = r'C:\Users\shaodunlin\OneDrive - Microsoft\VSCODE\4D_results_26_Jan_2026.csv'
OUTPUT_FILE = r'C:\Users\shaodunlin\OneDrive - Microsoft\VSCODE\Predict_4D_26_Jan_2026.md'
SEQUENCE_LENGTH = 20  # Double the sequence history
BATCH_SIZE = 128      # Larger batches for better parallelization
HIDDEN_SIZE = 2048    # Significantly larger hidden state
NUM_LAYERS = 4        # Deeper network
EPOCHS = 200          # More training
LEARNING_RATE = 0.0005 # Stabler learning for larger model
NUM_CLASSES = 10000 # 0000 to 9999
NUM_WORKERS = 0  # Windows safety: use 0 unless running under __main__
WEIGHT_DECAY = 0.0  # Lower training loss; higher overfit risk
DROPOUT = 0.0       # Lower training loss; higher overfit risk
MAX_GRAD_NORM = 1.0

# Create dataset for LSTM
# Input: Sequence of N 'draw vectors'. Each draw vector is a multi-hot encoding of size 10000.
# Output: The next draw vector (multi-hot).
class LotteryDataset(Dataset):
    def __init__(self, draw_matrix, seq_length):
        self.draw_matrix = draw_matrix
        self.seq_length = seq_length

    def __len__(self):
        return len(self.draw_matrix) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence
        x = self.draw_matrix[idx : idx + self.seq_length]
        # Target draw
        y = self.draw_matrix[idx + self.seq_length]
        return x, y

class LotteryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LotteryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # 1. Data Loading and Preprocessing
    print("Loading and preprocessing data...")
    df = pd.read_csv(DATA_FILE)

    # Ensure 'Number' is treated as string to preserve leading zeros, but we need int for index
    df['Number'] = df['Number'].apply(lambda x: int(str(x)))
    df['Draw Date'] = pd.to_datetime(df['Draw Date'])

    # Group by date to get all numbers for each draw
    # Sort by date ascending
    df_sorted = df.sort_values('Draw Date')
    draws = df_sorted.groupby('Draw Date')['Number'].apply(list).reset_index()

    # Convert to list of lists (each inner list is a draw's winning numbers)
    # Oldest first
    history_draws = draws['Number'].tolist()

    print(f"Total draws found: {len(history_draws)}")

    # Precompute multi-hot draw matrix once to avoid per-batch Python loops.
    print("Building draw matrix...")
    draw_matrix = torch.zeros(len(history_draws), NUM_CLASSES, dtype=torch.float32)
    for i, draw in enumerate(history_draws):
        if draw:
            draw_matrix[i, torch.tensor(draw, dtype=torch.long)] = 1.0

    dataset = LotteryDataset(draw_matrix, SEQUENCE_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # 2. Model Definition
    model = LotteryLSTM(NUM_CLASSES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss() # Use BCE for multi-label classification
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

# 3. Training
    print("Starting training...")
    model.train()
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        num_batches = len(dataloader)
        
        # Add progress bar for batches
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                # Update progress bar with loss info
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        # Print summary every epoch
        elapsed_total = time.time() - start_time
        if (epoch + 1) % 1 == 0:
            eta_seconds = (elapsed_total / (epoch + 1)) * (EPOCHS - epoch - 1)
            eta_minutes = eta_seconds / 60
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s | Elapsed: {elapsed_total/60:.1f}m | ETA: {eta_minutes:.1f}m")

# 4. Prediction
    print("Generating prediction...")
    model.eval()
    # Get the last SEQUENCE_LENGTH draws to predict the next one
    x_input = draw_matrix[-SEQUENCE_LENGTH:].unsqueeze(0).to(device)
    with torch.no_grad():
        prediction_logits = model(x_input)
        # Apply sigmoid to get probabilities
        prediction_probs = torch.sigmoid(prediction_logits).squeeze().cpu().numpy()

# Get top 23 numbers (standard draw size)
    top_indices = prediction_probs.argsort()[-23:][::-1]
    # Convert indices back to 4D strings
    predicted_numbers = [f"{num:04d}" for num in top_indices]

# Basic Statistics for the predicted numbers
    stats_summary = []
    for num_str in predicted_numbers:
        num_int = int(num_str)
        count = df[df['Number'] == num_int].shape[0]
        last_draw = df[df['Number'] == num_int]['Draw Date'].max()
        overdue_days = (pd.to_datetime('today') - last_draw).days if pd.notna(last_draw) else "Never"
        prob = prediction_probs[num_int]
        stats_summary.append(f"| {num_str} | {prob:.4f} | {count} | {overdue_days} days |")

# 5. Output Generation
    print(f"Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# Singapore 4D Prediction for {datetime.now().strftime('%d %b %Y')}\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Method:** LSTM Neural Network (PyTorch)\n")
        f.write(f"**Device Used:** {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}\n\n")
        
        f.write("> [!IMPORTANT]\n")
        f.write("> **Disclaimer:** These predictions are generated by a machine learning model based on historical patterns. Lottery draws are random independent events. Use this for entertainment or research purposes only.\n\n")

        f.write("## Top Predicted Numbers\n\n")
        f.write("The following numbers have the highest probability of appearing in the next draw, based on the learned sequences by the LSTM model.\n\n")
        
        f.write("| Number | Model Probability | Historical Frequency | Overdue (Days) |\n")
        f.write("| :---: | :---: | :---: | :---: |\n")
        for row in stats_summary:
            f.write(f"{row}\n")
        
        f.write("\n\n## Model Training Details\n")
        f.write(f"- **Epochs:** {EPOCHS}\n")
        f.write(f"- **Batch Size:** {BATCH_SIZE}\n")
        f.write(f"- **Sequence Length:** {SEQUENCE_LENGTH}\n")
        f.write(f"- **Loss Function:** Binary Cross Entropy\n")
        f.write(f"- **Hidden Layers:** {NUM_LAYERS} x {HIDDEN_SIZE}\n")

    print("Done!")

if __name__ == '__main__':
    main()
