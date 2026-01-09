"""
Neural Network for Matrix to Complex Number Mapping
"""

import os
import random
import warnings

# Suppress OpenMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

# ====================== Global Configuration ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")


# ====================== Data Loading Functions ======================
def load_data():
    """Load all necessary data files from data/ directory"""
    data_dir = 'data/'
    all_list_c = np.loadtxt(os.path.join(data_dir, 'all_list_c.txt'), delimiter=' ')
    all_matrix = np.loadtxt(os.path.join(data_dir, 'all_matrix.txt'), delimiter=' ').reshape(-1, 8, 8)
    s_re_c = np.loadtxt(os.path.join(data_dir, 's_re_c.txt'), delimiter=' ')

    return all_list_c, all_matrix, s_re_c


def build_mapping_dicts(X, X_c):
    """Build bidirectional mapping dictionaries between matrices and X_c values"""
    matrix_to_xc = {}
    xc_to_matrix = {}

    for i in range(len(X)):
        # Matrix -> X_c mapping
        matrix_key = X[i].astype(np.int8).tobytes()
        matrix_to_xc[matrix_key] = X_c[i]

        # X_c -> Matrix mapping
        xc_key = X_c[i].astype(np.int8).tobytes()
        xc_to_matrix[xc_key] = X[i]

    print(f"Mapping dictionaries built successfully. Total entries: {len(matrix_to_xc)}")
    return matrix_to_xc, xc_to_matrix


def get_xc_from_matrix(matrix, matrix_to_xc_dict):
    """
    Find corresponding 8 integers for given 8x8 matrix

    Args:
        matrix: 8x8 numpy array
        matrix_to_xc_dict: dictionary mapping matrices to X_c values

    Returns:
        Array of 8 integers, None if not found
    """
    matrix = matrix.reshape(8, 8)
    key = matrix.astype(np.int8).tobytes()
    return matrix_to_xc_dict.get(key, None)


# ====================== Dataset Splitting Functions ======================
def split_dataset(x, y, ratio=0.1, shuffle=True, uniform=False, batch_align=False,
                  batch_size=2, verbose=True):
    """
    Split dataset into training and validation sets

    Args:
        x: Input data
        y: Output labels
        ratio: Validation set ratio
        shuffle: Whether to shuffle data
        uniform: Whether to use uniform sampling
        batch_align: Whether to align data size for batch training
        batch_size: Batch size for alignment
        verbose: Whether to print split information

    Returns:
        (x_train, y_train), (x_val, y_val)
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f'Input [{x.shape[0]}] and output [{y.shape[0]}] data must have same number of samples')

    if ratio <= 0 or ratio >= 1:
        raise ValueError('Ratio must be between 0 and 1')

    val_size = int(ratio * x.shape[0])
    train_size = x.shape[0] - val_size

    # Adjust sizes for batch alignment if needed
    if batch_align and train_size > batch_size:
        val_size += train_size % batch_size
        train_size -= train_size % batch_size

    if shuffle:
        indices = np.random.permutation(x.shape[0])
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]
    else:
        if uniform:
            val_indices = np.arange(x.shape[0])[1:-1:(x.shape[0] // val_size)]
            train_mask = np.ones(x.shape[0], dtype=bool)
            train_mask[val_indices] = False
            train_indices = np.where(train_mask)[0]
            x_val, y_val = x[val_indices], y[val_indices]
            x_train, y_train = x[train_indices], y[train_indices]
        else:
            x_train, y_train = x[:train_size], y[:train_size]
            x_val, y_val = x[train_size:], y[train_size:]

    if verbose:
        print(f'Total samples: {x.shape[0]}, '
              f'Training: {x_train.shape[0]}, '
              f'Validation: {x_val.shape[0]}')

    return (x_train, y_train), (x_val, y_val)


def create_dataloaders(X, Y, val_ratio=0.25, batch_size=128, shuffle_train=True):
    """Create training and validation data loaders"""
    (X_train, Y_train), (X_val, Y_val) = split_dataset(
        X, Y, ratio=val_ratio, shuffle=True, verbose=True
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float().to(DEVICE),
        torch.from_numpy(Y_train).float().to(DEVICE)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)

    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float().to(DEVICE),
        torch.from_numpy(Y_val).float().to(DEVICE)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ====================== Model Components ======================
class ComplexDistanceLoss(nn.Module):
    """Custom complex number distance loss function"""

    def __init__(self):
        super(ComplexDistanceLoss, self).__init__()

    def forward(self, outputs, targets, return_elements=False):
        # Decompose outputs and targets
        e_real, e_imag, f_real, f_imag = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        e1_real, e1_imag, f1_real, f1_imag = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

        # Calculate differences
        e_real_diff = e_real - e1_real
        e_imag_diff = e_imag - e1_imag
        f_real_diff = f_real - f1_real
        f_imag_diff = f_imag - f1_imag

        # Calculate distances
        e_distance = torch.sqrt(e_real_diff ** 2 + e_imag_diff ** 2)
        f_distance = torch.sqrt(f_real_diff ** 2 + f_imag_diff ** 2)

        if return_elements:
            return e_distance, f_distance
        # Return mean distance
        return torch.mean(e_distance + f_distance)


class AttentionBlock(nn.Module):
    """Attention mechanism block"""

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # Compute attention weights
        attention = self.softmax(torch.bmm(query, key))

        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x


def make_residual_block(channels):
    """Create residual block"""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels)
    )


# ====================== Main Model ======================
class ResidualCNN(nn.Module):
    """Residual Convolutional Neural Network for matrix to complex mapping"""

    def __init__(self):
        super(ResidualCNN, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv2d(1, 64, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(64)

        # Residual block 1
        self.res_block1 = make_residual_block(64)

        # Attention mechanisms
        self.attention1 = AttentionBlock(64)
        self.attention2 = AttentionBlock(128)

        # Transition layer
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(128)

        # Residual block 2
        self.res_block2 = make_residual_block(128)

        # Pooling and fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 4)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input shape: [batch, 8, 8]
        x = x.unsqueeze(1)  # [batch, 1, 8, 8]

        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # [batch, 64, 8, 8]

        # Residual block 1
        residual = x
        x = self.res_block1(x)
        x += residual
        x = self.relu(x)

        # Attention 1
        x = self.attention1(x)

        # Transition layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)  # [batch, 128, 8, 8]

        # Residual block 2
        residual = x
        x = self.res_block2(x)
        x += residual
        x = self.relu(x)

        # Attention 2
        x = self.attention2(x)

        # Pooling
        x = self.adaptive_pool(x)  # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        x = self.tanh(x)  # [batch, 4]

        return x


# ====================== Training Functions ======================
def set_random_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, num_epochs=500,
                learning_rate=0.01, model_save_dir='models/'):
    """
    Train the neural network model

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        model_save_dir: Directory to save models

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    print("Starting training...")

    # Create save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)

    # Setup loss function and optimizer
    criterion = ComplexDistanceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=10,
        min_lr=1e-6,
        verbose=True
    )

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    # Move model to device
    model.to(DEVICE)

    # Training loop
    epochs_pbar = tqdm(range(num_epochs), desc='Training Progress')
    for epoch in epochs_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

            best_model_path = os.path.join(model_save_dir, 'best_random_tune_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'optimizer_state_dict': optimizer.state_dict()
            }, best_model_path)

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        epochs_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.6f}',
            'Val Loss': f'{avg_val_loss:.6f}',
            'Best Val Loss': f'{best_val_loss:.6f}',
            'LR': f'{current_lr:.2e}',
            'Best Epoch': best_epoch
        })

    print(f'Training completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}')

    # Save final model
    final_model_path = os.path.join(model_save_dir, 'last_random_tune_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': num_epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_val_loss': val_losses[-1]
    }, final_model_path)

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)

    return train_losses, val_losses


# ====================== Visualization Functions ======================
def plot_loss_curves(train_losses, val_losses, save_dir='outputs/'):
    """
    Plot training and validation loss curves

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot loss curves
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7, linewidth=2)

    # Find minimum validation loss
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss)

    # Mark minimum point
    plt.scatter(min_val_epoch, min_val_loss, color='darkred', s=100, zorder=5,
                label=f'Min Val Loss: {min_val_loss:.2e}')

    # Add annotation
    plt.annotate(f'Min: {min_val_loss:.2e}',
                 xy=(min_val_epoch, min_val_loss),
                 xytext=(min_val_epoch + 0.5, min_val_loss * 1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Configure plot
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(save_dir, 'training_loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Loss curve saved to {save_path}')

    # Print loss values
    print('\nTraining losses (first 10):')
    for i, loss in enumerate(train_losses[:10]):
        print(f'Epoch {i}: {loss:.4e}')

    print('\nValidation losses (first 10):')
    for i, loss in enumerate(val_losses[:10]):
        print(f'Epoch {i}: {loss:.4e}')

    plt.show()


# ====================== Testing Functions ======================
def test_model(model_path, test_loader, matrix_to_xc_dict, save_dir='outputs/'):
    """
    Evaluate model on test set and visualize results

    Args:
        model_path: Path to model checkpoint
        test_loader: Test data loader
        matrix_to_xc_dict: Dictionary mapping matrices to X_c values
        save_dir: Directory to save results

    Returns:
        Dictionary containing test metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = ResidualCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Testing model from: {model_path}")

    model.eval()
    model.to(DEVICE)

    # Collect test data
    all_inputs, all_targets, all_outputs = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)

            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    # Concatenate data
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # Get X_c values for all samples
    all_xc = []
    missing_count = 0

    for i in range(len(all_inputs)):
        input_matrix = all_inputs[i].reshape(8, 8)
        xc = get_xc_from_matrix(input_matrix, matrix_to_xc_dict)
        if xc is not None:
            all_xc.append(xc)
        else:
            all_xc.append(np.zeros(8))
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} samples have no corresponding X_c in dictionary")

    all_xc = np.array(all_xc)

    # Save results to CSV
    data_all = np.concatenate([all_xc, all_targets, all_outputs], axis=1)
    header = [
        "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
        "true_in_re", "true_in_im", "true_me_re", "true_me_im",
        "predict_in_re", "predict_in_im", "predict_me_re", "predict_me_im"
    ]

    csv_path = os.path.join(save_dir, 'result.csv')
    np.savetxt(
        csv_path,
        data_all,
        delimiter=",",
        header=",".join(header),
        comments=""
    )
    print(f"Results saved to {csv_path}")

    # Visualize results
    visualization_path = os.path.join(save_dir, 'model_test_results.png')
    visualize_predictions(all_inputs, all_targets, all_outputs, all_xc,
                          save_path=visualization_path)
    plt.show()
    # Calculate metrics
    metrics = calculate_test_metrics(all_targets, all_outputs)

    print("\n" + "=" * 60)
    print("Overall Test Results:")
    print("=" * 60)
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"E Distance (Complex 1): {metrics['e_distance']:.6f}")
    print(f"F Distance (Complex 2): {metrics['f_distance']:.6f}")
    print(f"Mean Distance Loss: {metrics['mean_distance']:.6f}")
    print(f"Number of samples: {len(all_targets)}")

    print("\nTesting completed!")

    return metrics


def visualize_predictions(all_inputs, all_targets, all_outputs, all_xc,
                          num_samples=10, save_path='outputs/predictions.png'):
    """Visualize model predictions"""
    # Create custom colormap
    colors = ["#add8e6", "#ffff00"]  # Light blue -> Yellow
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Randomly select samples
    num_samples = min(num_samples, len(all_inputs))
    random_indices = np.random.choice(len(all_inputs), num_samples, replace=False)

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Model Predictions vs Ground Truth', fontsize=16, y=1.02)

    sample_metrics = []

    for i, idx in enumerate(random_indices):
        # Get sample data
        input_sample = all_inputs[idx].reshape(8, 8)
        target_sample = all_targets[idx]
        output_sample = all_outputs[idx]
        xc_sample = all_xc[idx]

        # Calculate metrics
        metrics = calculate_sample_metrics(target_sample, output_sample)
        metrics.update({
            'index': idx,
            'target': target_sample,
            'output': output_sample,
            'x_c': xc_sample
        })
        sample_metrics.append(metrics)

        # Determine subplot position
        row = i // 5
        col = i % 5

        if row < 2:  # Only plot if we have space
            # Plot input matrix
            axes[row, col].imshow(input_sample, cmap=custom_cmap)

            # Create title with X_c values
            if np.any(xc_sample != 0):
                xc_int = [int(v) for v in xc_sample[:4]]  # Show first 4 values
                title = f'Sample {idx}\nX_c: {xc_int}...'
            else:
                title = f'Sample {idx}\nX_c: Not found'

            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

    # Print sample metrics
    print_sample_metrics(sample_metrics)

    return sample_metrics


def calculate_sample_metrics(target, output):
    """Calculate metrics for a single sample"""
    mse = np.mean((target - output) ** 2)
    mae = np.mean(np.abs(target - output))

    # Complex distance calculations
    e_real_diff = target[0] - output[0]
    e_imag_diff = target[1] - output[1]
    e_distance = (e_real_diff ** 2 + e_imag_diff ** 2) ** 0.5

    f_real_diff = target[2] - output[2]
    f_imag_diff = target[3] - output[3]
    f_distance = (f_real_diff ** 2 + f_imag_diff ** 2) ** 0.5

    distance_loss = (e_distance + f_distance) / 2

    return {
        'mse': mse,
        'mae': mae,
        'e_distance': e_distance,
        'f_distance': f_distance,
        'distance_loss': distance_loss
    }


def calculate_test_metrics(all_targets, all_outputs):
    """Calculate overall test metrics"""
    mse = np.mean((all_targets - all_outputs) ** 2)
    mae = np.mean(np.abs(all_targets - all_outputs))

    # Complex distances
    e_real_diff = all_targets[:, 0] - all_outputs[:, 0]
    e_imag_diff = all_targets[:, 1] - all_outputs[:, 1]
    e_distance = np.mean((e_real_diff ** 2 + e_imag_diff ** 2) ** 0.5)

    f_real_diff = all_targets[:, 2] - all_outputs[:, 2]
    f_imag_diff = all_targets[:, 3] - all_outputs[:, 3]
    f_distance = np.mean((f_real_diff ** 2 + f_imag_diff ** 2) ** 0.5)

    mean_distance = (e_distance + f_distance) / 2

    return {
        'mse': mse,
        'mae': mae,
        'e_distance': e_distance,
        'f_distance': f_distance,
        'mean_distance': mean_distance,
        'num_samples': len(all_targets)
    }


def print_sample_metrics(sample_metrics):
    """Print metrics for individual samples"""
    print("\n" + "=" * 60)
    print("Individual Sample Metrics:")
    print("=" * 60)

    for metrics in sample_metrics:
        idx = metrics['index']
        print(f"\nSample {idx}:")

        if np.any(metrics['x_c'] != 0):
            xc_int = [int(v) for v in metrics['x_c']]
            print(f"  X_c (8 integers): {xc_int}")
        else:
            print(f"  X_c (8 integers): Not found")

        print(f"  Target:     {metrics['target'].tolist()}")
        print(f"  Prediction: {metrics['output'].tolist()}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  E Distance: {metrics['e_distance']:.6f}")
        print(f"  F Distance: {metrics['f_distance']:.6f}")
        print(f"  Distance Loss: {metrics['distance_loss']:.6f}")


# ====================== Model Loading Function ======================
def load_trained_model(model_path):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = ResidualCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return model, checkpoint


# ====================== Main Function ======================
def main():
    """Main execution function"""
    # Set random seed for reproducibility
    set_random_seed(42)

    # Load data
    print("Loading data...")
    X_c, X, Y = load_data()
    print(f"X_c shape: {X_c.shape}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # Build mapping dictionaries
    matrix_to_xc, xc_to_matrix = build_mapping_dicts(X, X_c)

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        X, Y, val_ratio=0.25, batch_size=128
    )

    # Create and train model
    print("\nInitializing model...")
    model = ResidualCNN()

    # Train model
    # train_losses, val_losses = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     num_epochs=500,
    #     learning_rate=0.01,
    #     model_save_dir='models/'
    # )

    # Test best model
    print("\n" + "=" * 60)
    print("Testing Best Model:")
    print("=" * 60)

    best_model_path = 'models/best_random_tune_model.pth'
    print("\n1. Testing on training set:")
    train_metrics = test_model(
        best_model_path,
        train_loader,
        matrix_to_xc,
        save_dir='outputs/train_results/'
    )

    print("\n2. Testing on validation set:")
    val_metrics = test_model(
        best_model_path,
        val_loader,
        matrix_to_xc,
        save_dir='outputs/val_results/'
    )

    # Test final model
    print("\n" + "=" * 60)
    print("Testing Final Model:")
    print("=" * 60)

    final_model_path = 'models/last_random_tune_model.pth'
    print("\n1. Testing on training set:")
    test_model(
        final_model_path,
        train_loader,
        matrix_to_xc,
        save_dir='outputs/train_results_final/'
    )

    print("\n2. Testing on validation set:")
    test_model(
        final_model_path,
        val_loader,
        matrix_to_xc,
        save_dir='outputs/val_results_final/'
    )

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":

    main()
