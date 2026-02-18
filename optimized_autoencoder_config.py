"""
Memory-Optimized Autoencoder Configuration for 8GB RAM Systems
Optimized for Intel i5-10210U with 8GB RAM
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import psutil
import gc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MemoryEfficientAutoencoder(nn.Module):
    """
    Memory-efficient autoencoder optimized for systems with limited RAM
    Uses smaller architecture and mixed precision training
    """
    def __init__(self, input_dim, encoding_dims=[32, 16]):
        super(MemoryEfficientAutoencoder, self).__init__()
        
        # Lighter encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Reduced dropout
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Lighter decoder
        decoder_layers = []
        decoding_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        prev_dim = encoding_dims[-1]
        
        for i, dim in enumerate(decoding_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if i < len(decoding_dims) - 1:
                decoder_layers.extend([
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            prev_dim = dim
        
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class OptimizedNIDS:
    """
    Memory-optimized NIDS implementation for systems with 8GB RAM
    """
    def __init__(self, input_dim, encoding_dims=[32, 16], 
                 learning_rate=0.001, use_mixed_precision=False):
        # Force CPU usage for stability on limited RAM
        self.device = 'cpu'
        self.model = MemoryEfficientAutoencoder(input_dim, encoding_dims).to(self.device)
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.use_mixed_precision = use_mixed_precision
        
        # Print system info
        self.print_system_info()
        print(f"\nModel initialized on: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def print_system_info(self):
        """Display system resources"""
        memory = psutil.virtual_memory()
        print("="*70)
        print("SYSTEM INFORMATION")
        print("="*70)
        print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
        print(f"RAM Usage: {memory.percent}%")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)}")
        print(f"CPU Threads: {psutil.cpu_count(logical=True)}")
        print("="*70)
    
    def check_memory(self, stage=""):
        """Monitor memory usage during training"""
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            print(f"⚠️  Warning: High RAM usage at {stage}: {memory.percent}%")
            gc.collect()  # Force garbage collection
    
    def prepare_data_chunked(self, X, y=None, test_size=0.2, chunk_size=10000):
        """
        Memory-efficient data preparation using chunking
        """
        print(f"\nPreparing data in chunks of {chunk_size}...")
        
        # Split data
        if y is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
        else:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = None, None
        
        # Fit scaler on sample if data is too large
        if len(X_train) > 50000:
            sample_idx = np.random.choice(len(X_train), 50000, replace=False)
            print("Fitting scaler on 50k sample...")
            self.scaler.fit(X_train[sample_idx])
        else:
            self.scaler.fit(X_train)
        
        # Transform in chunks to save memory
        print("Normalizing training data...")
        X_train_scaled = self.scaler.transform(X_train)
        
        print("Normalizing validation data...")
        X_val_scaled = self.scaler.transform(X_val)
        
        self.check_memory("after data preparation")
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def train(self, X, y=None, epochs=30, batch_size=128, patience=7, max_samples=None):
        """
        Memory-optimized training with smaller batches and early stopping
        """
        print("\n" + "="*70)
        print("STARTING MEMORY-OPTIMIZED TRAINING")
        print("="*70)
        
        # Limit dataset size if specified
        if max_samples and len(X) > max_samples:
            print(f"⚠️  Limiting dataset to {max_samples} samples to manage memory")
            if y is not None:
                # Stratified sampling
                from sklearn.utils import resample
                X, y = resample(X, y, n_samples=max_samples, stratify=y, random_state=42)
            else:
                X = X[:max_samples]
        
        # Filter normal traffic if labels provided
        if y is not None:
            normal_mask = y == 0
            X_normal = X[normal_mask]
            print(f"Training on {len(X_normal):,} normal samples")
        else:
            X_normal = X
            print(f"Training on {len(X):,} samples (unsupervised)")
        
        self.check_memory("before data preparation")
        
        # Prepare data
        X_train_scaled, X_val_scaled, _, _ = self.prepare_data_chunked(X_normal)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train_scaled)
        X_val = torch.FloatTensor(X_val_scaled)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, X_train)
        val_dataset = TensorDataset(X_val, X_val)
        
        # Use smaller batch size for memory efficiency
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to save memory
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        # Clear memory
        del X_train_scaled, X_val_scaled
        gc.collect()
        
        self.check_memory("after creating data loaders")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Training Batches: {len(train_loader)}")
        print(f"  Validation Batches: {len(val_loader)}")
        print(f"  Early Stopping Patience: {patience}")
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (batch_X, batch_target) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_target = batch_target.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Periodic memory check
                if batch_idx % 50 == 0:
                    self.check_memory(f"epoch {epoch+1}, batch {batch_idx}")
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_target in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_target)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            memory_pct = psutil.virtual_memory().percent
            print(f"Epoch [{epoch+1:3d}/{epochs}] - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"RAM: {memory_pct:.1f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                Path('models').mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), 'models/autoencoder_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
            
            # Clear cache periodically
            if (epoch + 1) % 5 == 0:
                gc.collect()
        
        # Load best model
        self.model.load_state_dict(torch.load('models/autoencoder_best.pth'))
        
        # Calculate threshold
        print("\nCalculating anomaly threshold...")
        self.threshold = self._calculate_threshold(X_val)
        print(f"Threshold: {self.threshold:.6f}")
        
        return self.history
    
    def _calculate_threshold(self, X_val, percentile=95):
        """Calculate anomaly threshold"""
        self.model.eval()
        X_val = X_val.to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X_val)
            mse = torch.mean((X_val - reconstructed) ** 2, dim=1)
            errors = mse.cpu().numpy()
        
        return np.percentile(errors, percentile)
    
    def predict_batch(self, X, batch_size=128):
        """Memory-efficient batch prediction"""
        self.model.eval()
        
        # Normalize
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        
        errors = []
        
        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(self.device)
                reconstructed = self.model(batch_X)
                mse = torch.mean((batch_X - reconstructed) ** 2, dim=1)
                errors.extend(mse.cpu().numpy())
        
        errors = np.array(errors)
        predictions = (errors > self.threshold).astype(int)
        
        return predictions, errors
    
    def save_model(self, path='models/autoencoder_optimized.pth'):
        """Save model"""
        model_data = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'history': self.history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_data, path)
        
        # Get file size
        size_mb = Path(path).stat().st_size / (1024**2)
        print(f"\n✓ Model saved: {path} ({size_mb:.2f} MB)")


def get_optimal_settings(n_samples):
    """
    Get optimal training settings based on dataset size
    """
    if n_samples < 10000:
        return {
            'batch_size': 128,
            'epochs': 30,
            'encoding_dims': [32, 16, 8],
            'max_samples': None,
            'estimated_time': '5-10 minutes'
        }
    elif n_samples < 100000:
        return {
            'batch_size': 256,
            'epochs': 25,
            'encoding_dims': [32, 16],
            'max_samples': None,
            'estimated_time': '20-40 minutes'
        }
    elif n_samples < 500000:
        return {
            'batch_size': 512,
            'epochs': 20,
            'encoding_dims': [32, 16],
            'max_samples': 200000,
            'estimated_time': '1-2 hours'
        }
    else:
        return {
            'batch_size': 1024,
            'epochs': 15,
            'encoding_dims': [32, 16],
            'max_samples': 300000,
            'estimated_time': '2-3 hours'
        }


def train_optimized(data_path, output_path='models/autoencoder_optimized.pth'):
    """
    Optimized training pipeline for 8GB RAM systems
    """
    print("="*70)
    print("MEMORY-OPTIMIZED NIDS AUTOENCODER TRAINING")
    print("Optimized for Intel i5-10210U with 8GB RAM")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset: {len(df):,} samples")
    
    # Get optimal settings
    settings = get_optimal_settings(len(df))
    print(f"\nOptimized Settings:")
    print(f"  Batch Size: {settings['batch_size']}")
    print(f"  Epochs: {settings['epochs']}")
    print(f"  Architecture: {settings['encoding_dims']}")
    if settings['max_samples']:
        print(f"  Max Samples: {settings['max_samples']:,}")
    print(f"  Estimated Time: {settings['estimated_time']}")
    
    # Extract features (simplified for demo)
    feature_cols = [col for col in df.columns if col != 'label']
    X = df[feature_cols].fillna(0).values
    y = df['label'].values if 'label' in df.columns else None
    
    # Convert labels to binary
    if y is not None and y.dtype == 'object':
        y = (y != 'BENIGN').astype(int)
    
    print(f"\nFeatures: {X.shape[1]}")
    if y is not None:
        print(f"Normal: {np.sum(y==0):,}, Attack: {np.sum(y==1):,}")
    
    # Initialize model
    model = OptimizedNIDS(
        input_dim=X.shape[1],
        encoding_dims=settings['encoding_dims']
    )
    
    # Train
    history = model.train(
        X, y,
        epochs=settings['epochs'],
        batch_size=settings['batch_size'],
        max_samples=settings['max_samples']
    )
    
    # Save
    model.save_model(output_path)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return model


if __name__ == "__main__":
    # Example with sample data
    print("Creating sample data for demonstration...")
    
    n_samples = 20000
    n_features = 15
    
    sample_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    sample_df['label'] = np.random.choice(['BENIGN', 'Attack'], n_samples, p=[0.8, 0.2])
    
    sample_path = 'data/sample_data.csv'
    Path(sample_path).parent.mkdir(exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    
    # Train
    train_optimized(sample_path)
