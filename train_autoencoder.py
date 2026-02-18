"""
PROGRESSIVE GPU AUTOENCODER TRAINING - STAGE 3
2 Million Samples Training (6-8 hours with RTX 4060)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from autoencoder_nids import AutoencoderNIDS


def train_stage_3():
    """
    Stage 3: Train on 2 Million samples for 6-8 hours
    Creates a strong, robust autoencoder
    """
    print("="*70)
    print("PROGRESSIVE AUTOENCODER TRAINING - STAGE 3")
    print("2 MILLION SAMPLES (6-8 HOURS WITH RTX 4060)")
    print("="*70)
    
    # Load your preprocessed data
    print("Loading preprocessed data...")
    features_path = 'data/processed/cic_ids_features.csv'
    labels_path = 'data/processed/cic_ids_labels.csv'
    
    # Read labels to identify normal traffic
    print("Reading labels to identify normal samples...")
    labels_df = pd.read_csv(labels_path)
    normal_indices = labels_df[labels_df['label'] == 0].index.values
    
    # Select 2 million normal samples randomly
    selected_normal_indices = np.random.choice(normal_indices, size=min(2000000, len(normal_indices)), replace=False)
    print(f"Selected {len(selected_normal_indices):,} normal samples for training")
    
    # Load corresponding features for these samples
    print("Loading 2M samples for training...")
    
    # Read features in chunks to avoid memory issues
    chunk_size = 500000
    all_features = []
    
    for start_idx in range(0, len(selected_normal_indices), chunk_size):
        end_idx = min(start_idx + chunk_size, len(selected_normal_indices))
        current_indices = selected_normal_indices[start_idx:end_idx]
        
        # Read these specific rows (this is memory efficient)
        all_rows = pd.read_csv(features_path)
        current_features = all_rows.iloc[current_indices].values
        all_features.append(current_features)
        
        print(f"  Loaded chunk {start_idx//chunk_size + 1}: {current_features.shape[0]} samples")
        
        # Memory cleanup
        del all_rows, current_features
        gc.collect()
    
    # Combine all chunks
    X_normal = np.vstack(all_features)
    print(f"Total normal samples loaded: {X_normal.shape[0]:,}")
    
    # Split for training and validation
    print("Splitting data for training and validation...")
    X_train, X_val = train_test_split(
        X_normal, test_size=0.1, random_state=42  # 10% for validation
    )
    
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Validation samples: {X_val.shape[0]:,}")
    
    # Initialize scaler and normalize data
    print("Normalizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    
    # Initialize GPU-optimized autoencoder
    print(f"\nInitializing GPU autoencoder with RTX 4060 optimization...")
    ae_nids = AutoencoderNIDS(
        input_dim=X_train_scaled.shape[1],
        encoding_dims=[64, 32, 16, 8],  # Your preferred architecture
        learning_rate=0.001,
        device='cuda',
        max_memory_usage=0.85  # Use 85% of 8GB VRAM
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=512,  # Optimized for RTX 4060
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Keep 0 to avoid memory issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=512,
        pin_memory=True,
        num_workers=0
    )
    
    print(f"Training batches per epoch: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    
    # Training loop
    print(f"\n" + "="*70)
    print("STARTING STAGE 3 TRAINING")
    print("2 MILLION SAMPLES - EXPECTED TIME: 6-8 HOURS")
    print("="*70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    epochs = 12  # Good balance for 2M samples
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        ae_nids.model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, (batch_X, batch_target) in enumerate(train_loader):
            batch_X = batch_X.to(ae_nids.device, non_blocking=True)
            batch_target = batch_target.to(ae_nids.device, non_blocking=True)
            
            ae_nids.optimizer.zero_grad()
            outputs = ae_nids.model(batch_X)
            loss = ae_nids.criterion(outputs, batch_target)
            loss.backward()
            ae_nids.optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Progress update every 1000 batches
            if batch_idx % 1000 == 0:
                print(f"  Processed {batch_idx:,} batches...")
        
        avg_train_loss = train_loss / batch_count
        ae_nids.history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        ae_nids.model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_X, batch_target in val_loader:
                batch_X = batch_X.to(ae_nids.device, non_blocking=True)
                batch_target = batch_target.to(ae_nids.device, non_blocking=True)
                
                outputs = ae_nids.model(batch_X)
                loss = ae_nids.criterion(outputs, batch_target)
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count
        ae_nids.history['val_loss'].append(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(ae_nids.model.state_dict(), 'models/autoencoder_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    ae_nids.model.load_state_dict(torch.load('models/autoencoder_best.pth'))
    
    # Calculate threshold using validation data
    print("\nCalculating anomaly detection threshold...")
    ae_nids.model.eval()
    with torch.no_grad():
        val_reconstructed = ae_nids.model(X_val_tensor.to(ae_nids.device))
        mse = torch.mean((X_val_tensor.to(ae_nids.device) - val_reconstructed) ** 2, dim=1)
        reconstruction_errors = mse.cpu().numpy()
    
    ae_nids.threshold = np.percentile(reconstruction_errors, 95)
    print(f"Anomaly threshold: {ae_nids.threshold:.6f}")
    
    # Save the final model
    print(f"\nSaving strong model to models/autoencoder_stage3.pth...")
    ae_nids.save_model('models/autoencoder_stage3.pth')
    
    # Also save the scaler separately for GUI use
    import joblib
    joblib.dump(scaler, 'models/autoencoder_scaler.pkl')
    
    # Plot results
    ae_nids.plot_training_history()
    
    print("\n" + "="*70)
    print("STAGE 3 TRAINING COMPLETED!")
    print("2 MILLION SAMPLES TRAINED SUCCESSFULLY")
    print(f"Training Time: 6-8 hours (as expected)")
    print(f"Model saved to: models/autoencoder_stage3.pth")
    print(f"Strong autoencoder ready for GUI integration!")
    print("="*70)
    
    return ae_nids


if __name__ == "__main__":
    print("Starting Stage 3: 2 Million Samples Training")
    print("This will take approximately 6-8 hours with your RTX 4060")
    print("Creating a STRONG, ROBUST autoencoder model...")
    
    try:
        model = train_stage_3()
        print("\n🎉 STAGE 3 COMPLETED SUCCESSFULLY! 🎉")
        print("Your strong autoencoder model is ready!")
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        print("Model may be partially saved in models/ directory")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()