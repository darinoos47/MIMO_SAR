"""
Comparison Script: End-to-End vs. Two-Stage Training

This script helps you run and compare different training strategies:
1. End-to-End Training
2. Two-Stage Training (Frozen Denoiser)
3. Two-Stage Training (Fine-tuned Denoiser)

Usage:
    python compare_training_strategies.py
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a shell command and print results."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False
    else:
        print(f"\n✅ SUCCESS: {description}")
        print(f"   Time: {elapsed_time:.2f} seconds")
        return True


def modify_config_file(filepath, modifications):
    """Modify configuration values in a Python file."""
    print(f"\nModifying {filepath}...")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        modified = False
        for key, value in modifications.items():
            if line.strip().startswith(key):
                # Replace the line
                indent = len(line) - len(line.lstrip())
                modified_lines.append(' ' * indent + f"{key} = {value}\n")
                modified = True
                print(f"  Set {key} = {value}")
                break
        
        if not modified:
            modified_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(modified_lines)
    
    print(f"✓ {filepath} modified")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        MIMO SAR Training Strategy Comparison                     ║
║                                                                  ║
║  This script will train the network using different strategies  ║
║  and save results for comparison.                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    NUM_EPOCHS_DENOISER = 50  # Reduced for quick comparison
    NUM_EPOCHS_FULL = 100     # Reduced for quick comparison
    
    print(f"\nConfiguration:")
    print(f"  Denoiser Epochs: {NUM_EPOCHS_DENOISER}")
    print(f"  Full Network Epochs: {NUM_EPOCHS_FULL}")
    print(f"\n(Edit this script to change epoch counts)")
    
    input("\nPress Enter to start, or Ctrl+C to cancel...")
    
    results = []
    
    # =========================================================================
    # Strategy 1: End-to-End Training
    # =========================================================================
    print("\n" + "="*70)
    print("STRATEGY 1: END-TO-END TRAINING")
    print("="*70)
    
    modify_config_file('train_configurable.py', {
        'NUM_EPOCHS': str(NUM_EPOCHS_FULL),
        'TRAINING_STRATEGY': "'end_to_end'",
        'TRAINING_MODE': "'unsupervised'",
    })
    
    success = run_command(
        'python train_configurable.py',
        'Strategy 1: End-to-End Training'
    )
    
    if success and os.path.exists('../../checkpoints/dbp_model.pth'):
        os.rename('../../checkpoints/dbp_model.pth', 'dbp_model_end_to_end.pth')
        print("✓ Saved model as: dbp_model_end_to_end.pth")
        results.append(('End-to-End', '✅ Success'))
    else:
        results.append(('End-to-End', '❌ Failed'))
    
    # =========================================================================
    # Strategy 2: Two-Stage Training (Frozen Denoiser)
    # =========================================================================
    print("\n" + "="*70)
    print("STRATEGY 2: TWO-STAGE TRAINING (FROZEN DENOISER)")
    print("="*70)
    
    # Stage 1: Train Denoiser
    print("\n--- Stage 1: Training Denoiser ---")
    modify_config_file('train_denoiser_only.py', {
        'NUM_EPOCHS': str(NUM_EPOCHS_DENOISER),
        'DENOISER_TRAINING_MODE': "'unsupervised'",
    })
    
    success_stage1 = run_command(
        'python train_denoiser_only.py',
        'Stage 1: Train Denoiser (Unsupervised)'
    )
    
    if not success_stage1 or not os.path.exists('../../checkpoints/denoiser_pretrained.pth'):
        print("❌ Stage 1 failed, skipping Stage 2")
        results.append(('Two-Stage (Frozen)', '❌ Failed at Stage 1'))
    else:
        # Stage 2: Train Full Network with Frozen Denoiser
        print("\n--- Stage 2: Training Full Network (Frozen Denoiser) ---")
        modify_config_file('train_configurable.py', {
            'NUM_EPOCHS': str(NUM_EPOCHS_FULL),
            'TRAINING_STRATEGY': "'two_stage'",
            'TRAINING_MODE': "'unsupervised'",
            'FREEZE_DENOISER': 'True',
        })
        
        success_stage2 = run_command(
            'python train_configurable.py',
            'Stage 2: Train Full Network (Frozen Denoiser)'
        )
        
        if success_stage2 and os.path.exists('../../checkpoints/dbp_model.pth'):
            os.rename('../../checkpoints/dbp_model.pth', 'dbp_model_two_stage_frozen.pth')
            print("✓ Saved model as: dbp_model_two_stage_frozen.pth")
            results.append(('Two-Stage (Frozen)', '✅ Success'))
        else:
            results.append(('Two-Stage (Frozen)', '❌ Failed at Stage 2'))
    
    # =========================================================================
    # Strategy 3: Two-Stage Training (Fine-tuned Denoiser)
    # =========================================================================
    print("\n" + "="*70)
    print("STRATEGY 3: TWO-STAGE TRAINING (FINE-TUNED DENOISER)")
    print("="*70)
    
    # Reuse denoiser from Strategy 2
    if not os.path.exists('../../checkpoints/denoiser_pretrained.pth'):
        print("❌ Pre-trained denoiser not found, skipping Strategy 3")
        results.append(('Two-Stage (Fine-tuned)', '❌ No pre-trained denoiser'))
    else:
        modify_config_file('train_configurable.py', {
            'NUM_EPOCHS': str(NUM_EPOCHS_FULL),
            'TRAINING_STRATEGY': "'two_stage'",
            'TRAINING_MODE': "'unsupervised'",
            'FREEZE_DENOISER': 'False',
        })
        
        success = run_command(
            'python train_configurable.py',
            'Strategy 3: Two-Stage Training (Fine-tuned Denoiser)'
        )
        
        if success and os.path.exists('../../checkpoints/dbp_model.pth'):
            os.rename('../../checkpoints/dbp_model.pth', 'dbp_model_two_stage_finetuned.pth')
            print("✓ Saved model as: dbp_model_two_stage_finetuned.pth")
            results.append(('Two-Stage (Fine-tuned)', '✅ Success'))
        else:
            results.append(('Two-Stage (Fine-tuned)', '❌ Failed'))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    
    print("\nResults:")
    for strategy, status in results:
        print(f"  {strategy:30s}: {status}")
    
    print("\nSaved Models:")
    models = [
        'dbp_model_end_to_end.pth',
        'dbp_model_two_stage_frozen.pth',
        'dbp_model_two_stage_finetuned.pth',
    ]
    for model in models:
        if os.path.exists(model):
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} (not created)")
    
    print("\nNext Steps:")
    print("  1. Test models using: python test_unsupervised_model.py")
    print("  2. Evaluate on full dataset: python test_full_dataset.py")
    print("  3. Compare visualizations in generated PNG files")
    print("  4. Analyze training loss curves")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

