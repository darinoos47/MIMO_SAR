"""
Quick comparison script to train with different modes and compare results.
Run this to see how supervised, unsupervised, and hybrid modes differ.
"""

import subprocess
import os
import shutil

def modify_config(mode, sup_weight=1.0, unsup_weight=1.0, epochs=50):
    """Modify training configuration in train_configurable.py"""
    with open('train_configurable.py', 'r') as f:
        lines = f.readlines()
    
    with open('train_configurable.py', 'w') as f:
        for line in lines:
            if line.startswith('TRAINING_MODE ='):
                f.write(f"TRAINING_MODE = '{mode}'  # Auto-set by compare script\n")
            elif line.startswith('SUPERVISED_WEIGHT ='):
                f.write(f"SUPERVISED_WEIGHT = {sup_weight}   # Auto-set by compare script\n")
            elif line.startswith('UNSUPERVISED_WEIGHT ='):
                f.write(f"UNSUPERVISED_WEIGHT = {unsup_weight} # Auto-set by compare script\n")
            elif line.startswith('NUM_EPOCHS ='):
                f.write(f"NUM_EPOCHS = {epochs}  # Auto-set by compare script\n")
            else:
                f.write(line)

def run_training(mode_name):
    """Run training and save results"""
    print(f"\n{'='*60}")
    print(f"Training with {mode_name.upper()} mode")
    print(f"{'='*60}\n")
    
    result = subprocess.run(['python', 'train_configurable.py'], 
                          capture_output=False, text=True)
    
    if result.returncode == 0:
        # Save model and plots with mode-specific names
        if os.path.exists('../../checkpoints/dbp_model.pth'):
            shutil.copy('../../checkpoints/dbp_model.pth', f'dbp_model_{mode_name}.pth')
        if os.path.exists(f'training_loss_{mode_name}.png'):
            # Already has the right name
            pass
        if os.path.exists('train_debug_plot_x.png'):
            shutil.copy('train_debug_plot_x.png', f'train_debug_plot_x_{mode_name}.png')
        if os.path.exists('train_debug_plot_y.png'):
            shutil.copy('train_debug_plot_y.png', f'train_debug_plot_y_{mode_name}.png')
        
        print(f"\n✓ {mode_name.upper()} training complete!")
        print(f"  Model saved: dbp_model_{mode_name}.pth")
    else:
        print(f"\n✗ {mode_name.upper()} training failed!")

def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║       MIMO SAR Training Mode Comparison                       ║
║                                                               ║
║  This script will train the network with three different     ║
║  modes and save results for comparison:                      ║
║                                                               ║
║  1. SUPERVISED:    Uses ground truth x (image domain)        ║
║  2. UNSUPERVISED:  Uses only measurements y                  ║
║  3. HYBRID:        Combines both losses                      ║
║                                                               ║
║  Each training will run for 50 epochs (quick comparison)     ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    backup_exists = os.path.exists('train_configurable.py.backup')
    if not backup_exists:
        # Backup original config
        shutil.copy('train_configurable.py', 'train_configurable.py.backup')
        print("✓ Backed up original configuration")
    
    try:
        # 1. Supervised mode
        modify_config(mode='supervised', sup_weight=1.0, unsup_weight=1.0, epochs=50)
        run_training('supervised')
        
        # 2. Unsupervised mode
        modify_config(mode='unsupervised', sup_weight=1.0, unsup_weight=1.0, epochs=50)
        run_training('unsupervised')
        
        # 3. Hybrid mode
        modify_config(mode='hybrid', sup_weight=1.0, unsup_weight=1.0, epochs=50)
        run_training('hybrid')
        
        print(f"\n{'='*60}")
        print("ALL TRAINING COMPLETE!")
        print(f"{'='*60}")
        print("\nResults saved:")
        print("  Models:      dbp_model_supervised.pth")
        print("               dbp_model_unsupervised.pth")
        print("               dbp_model_hybrid.pth")
        print("\n  Loss curves: training_loss_supervised.png")
        print("               training_loss_unsupervised.png")
        print("               training_loss_hybrid.png")
        print("\n  Debug plots: train_debug_plot_x_{mode}.png")
        print("               train_debug_plot_y_{mode}.png")
        print("\nCompare the loss curves and debug plots to see the differences!")
        
    finally:
        # Restore original config
        if os.path.exists('train_configurable.py.backup'):
            shutil.copy('train_configurable.py.backup', 'train_configurable.py')
            print("\n✓ Restored original configuration")

if __name__ == '__main__':
    main()

