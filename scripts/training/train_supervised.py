import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import time

# Add project root to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


from core.data_loader import MIMOSAR_Dataset
from core.models import DBPNet
from core.utils import complex_matmul

# -----------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------
MAT_FILE = '../../data/FL_MIMO_SAR_data.mat'
MODEL_SAVE_PATH = '../../checkpoints/dbp_model.pth'

# Training Hyperparameters
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# Model Hyperparameters
NUM_UNROLLS = 1     # N1 in the paper: total unrolled iterations
NUM_ADMM_STEPS = 1   # N2 in the paper: internal ADMM steps

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
    
    tic = time.time()
    
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

    toc = time.time()
    print(f"Total Training time is: {toc-tic} seconds")

    # Debug code - start
    print("Generating debug plots of the last training batch...")
    with torch.no_grad():
        # --- PLOT 1: Measurement Domain (y vs y_hat) ---

        # Select the first sample (index 0) from the last batch
        y_sample = y_batch[0]       # Shape [2, N_v]
        y_hat_sample = y_hat_batch[0] # Shape [2, N_v]

        # Move to CPU, get the real part (index 0), and convert to numpy
        y_real = y_sample[0].cpu().numpy()
        y_hat_real = y_hat_sample[0].cpu().numpy()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_real, 'r', label='Ground Truth (y_batch)[0, real]')
        plt.plot(y_hat_real, 'b--', label='Network Output (y_hat_batch)[0, real]')
        plt.title(f'Debug Plot - Last Batch (Epoch {NUM_EPOCHS}) - Measurement Domain')
        plt.xlabel('Virtual Antenna Index')
        plt.ylabel('Amplitude (Real Part)')
        plt.legend()
        plt.grid(True)

        debug_plot_path_y = 'train_debug_plot_y.png'
        plt.savefig(debug_plot_path_y)
        print(f"Measurement debug plot saved to {debug_plot_path_y}")
        plt.close() # Close the figure to start the next one

        # --- PLOT 2: Image Domain (x_hat) ---

        # Select the corresponding x_hat sample (index 0)
        x_hat_sample = x_hat_batch[0] # Shape [2, N_theta]

        # Get real part and convert to numpy
        x_hat_real = x_hat_sample[0].cpu().numpy()

        # *** FIX: Get NUM_ANGLE_BINS from actual data instead of hardcoding ***
        NUM_ANGLE_BINS = x_hat_real.shape[0]  # Dynamically get from data
        START_ANGLE_DEG = 25.0
        END_ANGLE_DEG = -25.0
        
        # Create the angle axis
        theta = np.linspace(START_ANGLE_DEG, END_ANGLE_DEG, NUM_ANGLE_BINS)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(theta, x_hat_real, 'b', label='Network Output (x_hat_batch)[0, real]')
        plt.title(f'Debug Plot - Last Batch (Epoch {NUM_EPOCHS}) - Image Domain')
        plt.xlabel('Angle (degree)')
        plt.ylabel('Amplitude (Real Part)')
        plt.legend()
        plt.grid(True)

        debug_plot_path_x = 'train_debug_plot_x.png'
        plt.savefig(debug_plot_path_x)
        print(f"Image debug plot saved to {debug_plot_path_x}")
        plt.close()

    # Debug code - end

    # -----------------------------------------------------------------
    # 6. Save the Trained Model
    # -----------------------------------------------------------------
    print("--- Training Complete ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
