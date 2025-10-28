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
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# Model Hyperparameters
NUM_UNROLLS = 5      # N1 in the paper: total unrolled iterations
NUM_ADMM_STEPS = 3   # N2 in the paper: internal ADMM steps

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
    # Add a 'batch' dimension for our matmul functions
    A_batch_tensor = A_tensor.unsqueeze(0) 

    # -----------------------------------------------------------------
    # 4. Initialize Model, Loss, and Optimizer
    # -----------------------------------------------------------------
    print("Initializing model...")
    model = DBPNet(
        A_tensor, 
        num_iterations=NUM_UNROLLS, 
        N_admm_steps=NUM_ADMM_STEPS
    ).to(device)

    # [cite_start]Loss function (in measurement domain, as per paper [cite: 581-584, 1137-1138])
    # This is the Euclidean norm loss: L = ||y_hat - y||_2^2
    criterion = nn.MSELoss()
    
    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model Parameters:\n  Num Unrolls (N1): {NUM_UNROLLS}\n  Num ADMM Steps (N2): {NUM_ADMM_STEPS}")

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
            
            # 2. Project x_hat back to measurement domain
            # y_hat = A * x_hat
            
            # <<< FIX WAS HERE: Use A_batch_tensor (4D) not A_tensor (3D) >>>
            y_hat_batch = complex_matmul(A_batch_tensor, x_hat_batch)
            
            # 3. Compute the unsupervised loss in the measurement domain
            # Loss = || y_hat - y ||^2
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
