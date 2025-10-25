import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from data_loader import MIMOSAR_Dataset
from models import DBPNet
from utils import complex_matmul

# -----------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------
MAT_FILE = 'FL_MIMO_SAR_data.mat'
MODEL_SAVE_PATH = 'dbp_model.pth'

# Training Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_DBP_ITERATIONS = 5 # Number of unrolled iterations in the network

# -----------------------------------------------------------------
# 2. Setup
# -----------------------------------------------------------------
def main():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    # -----------------------------------------------------------------
    # 3. Load Data
    # -----------------------------------------------------------------
    print("Loading dataset...")
    dataset = MIMOSAR_Dataset(MAT_FILE)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, # Use 2 worker processes for data loading
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Get steering matrix A and move it to the device
    A_tensor = dataset.A.to(device)

    # -----------------------------------------------------------------
    # 4. Initialize Model, Loss, and Optimizer
    # -----------------------------------------------------------------
    print("Initializing model...")
    model = DBPNet(A_tensor, num_iterations=NUM_DBP_ITERATIONS).to(device)

    # Loss function (in measurement domain, as per paper )
    criterion = nn.MSELoss()
    
    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------------------------------------------
    # 5. Training Loop
    # -----------------------------------------------------------------
    print(f"--- Starting Unsupervised Training for {NUM_EPOCHS} Epochs ---")
    
    model.train() # Set the model to training mode
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        
        for i, y_batch in enumerate(dataloader):
            # Move data to the selected device
            y_batch = y_batch.to(device)
            
            # --- Forward Pass ---
            # 1. Get the estimated reflectivity x_hat from the model
            x_hat_batch = model(y_batch)
            
            # 2. Project x_hat back to measurement domain [cite: 583]
            # y_hat = A * x_hat
            y_hat_batch = complex_matmul(A_tensor, x_hat_batch)
            
            # 3. Compute the unsupervised loss in the measurement domain [cite: 583]
            # Loss = || y - y_hat ||^2
            loss = criterion(y_hat_batch, y_batch)
            
            # --- Backward Pass and Optimization ---
            # 1. Clear previous gradients
            optimizer.zero_grad()
            
            # 2. Compute gradients
            loss.backward()
            
            # 3. Update model parameters
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")

        # Print average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"*** Epoch {epoch+1} Complete. Average Loss: {avg_epoch_loss:.6f} ***")

    # -----------------------------------------------------------------
    # 6. Save the Trained Model
    # -----------------------------------------------------------------
    print("--- Training Complete ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
