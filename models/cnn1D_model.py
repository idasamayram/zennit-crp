import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold



# ------------------------
# 1️⃣ Custom Dataset Class
# ------------------------

class VibrationDataset(Dataset):
    '''
    This version includes the operation data so that it can be used for stratified
    sampling in the train/val/test split.
    '''
    def __init__(self, data_dir, augment_bad=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.operations = []  # Optional for operation-based stratification
        self.augment_bad = augment_bad
        self.file_groups = []  # e.g., 'M01_Feb_2019_OP02_000'

        for label, label_idx in zip(["good", "bad"], [0, 1]):  # 0=good, 1=bad
            folder = self.data_dir / label
            for file_name in folder.glob("*.h5"):
                self.file_paths.append(file_name)
                self.labels.append(label_idx)
                # Extract operation (e.g., 'OP02' from 'M01_Feb_2019_OP02_000_window_0.h5')
                operation = file_name.stem.split('_')[3]
                self.operations.append(operation)
                # Extract file group (e.g., 'M01_Feb_2019_OP02_000')
                file_group = file_name.stem.rsplit('_window_', 1)[0]
                self.file_groups.append(file_group)

        self.labels = np.array(self.labels)
        self.operations = np.array(self.operations)
        self.file_groups = np.array(self.file_groups)
        assert len(self.file_paths) == 7501, f"Expected 7501 files, found {len(self.file_paths)}"

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3)

        data = np.transpose(data, (1, 0))  # Change to (3, 2000) for CNN

        label = self.labels[idx]

        # Augment bad samples by adding noise
        if self.augment_bad and label == 1:
            data += np.random.normal(0, 0.01, data.shape)  # Add Gaussian noise

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
# ------------------------
# ------------------------

# 2️⃣ Define the CNN Model for downsampled data
# ------------------------
class CNN1D_DS(nn.Module):
    def __init__(self):
        super(CNN1D_DS, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=9, stride=1)
        self.gn1 = nn.GroupNorm(4, 16)  # GroupNorm replaces BatchNorm
        #self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(0.2)  # Add dropout after conv1

        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=1)
        self.gn2 = nn.GroupNorm(4, 32)
        #self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(0.2)  # Add dropout after conv2

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
        self.gn3 = nn.GroupNorm(4, 64)
        #self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.dropout3 = nn.Dropout(0.2)  # Add dropout after conv3

        #changed this part compare to cnn1D_torch_update_2 which flattened the layer

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.3)  # Increased dropout to reduce overfitting
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.gn1(self.conv1(x))))
        x = self.pool2(self.relu(self.gn2(self.conv2(x))))
        x = self.pool3(self.relu(self.gn3(self.conv3(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (we use CrossEntropyLoss)

        return x


class CNN1D_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_Wide, self).__init__()
        # Wider kernels to increase receptive field
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)  # Increased kernel size
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout1 = nn.Dropout(0.2)  # Add dropout after first layer

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)  # Increased kernel size
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout2 = nn.Dropout(0.2)  # Add dropout after second layer

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)  # Increased kernel size
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout3 = nn.Dropout(0.2)  # Add dropout after third layer

        # NEW: Add a fourth convolutional layer for deeper network
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)  # Changed input size to match conv4 output
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.4)  # Increased dropout for final layer
        self.relu = nn.LeakyReLU(0.1)  # Using LeakyReLU for better gradient flow

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(self.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(self.relu(self.conv4(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (we use CrossEntropyLoss)

        return x

class CNN_1d(nn.Module):
    def __init__(self, dropout=0.3, n_out=2):
        super(CNN_1d, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=9, stride=1, padding=4),  # Reduce filters, increase kernel size
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout)
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),  # Add intermediate FC layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_out)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x




# ------------------------
# 3️⃣ Train & Evaluate Functions
# ------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy



def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    misclassified_indices = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect indices of misclassified samples
            batch_indices = torch.where(predicted != labels)[0]
            for idx in batch_indices:
                global_idx = batch_idx * val_loader.batch_size + idx.item()
                misclassified_indices.append(global_idx)

    val_loss /= len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc, misclassified_indices

# ------------------------
# 4️⃣ Test the Model
# ------------------------
def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute F1-score
    f1 = f1_score(all_labels, all_preds, average="weighted")
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    return f1, accuracy


# ------------------------
# 5️⃣ Full Training Pipeline
# ------------------------
def train_and_evaluate(train_loader, val_loader, test_loader, epochs=20, lr=0.001, weight_decay=1e-4, EralyStopping=False, Schedule=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Model setup
    model = CNN1D_Wide().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    early_stop_epoch = epochs
    patience = 3


    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # or use ReduceLROnPlateau scheduler
    # scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  factor=0.5,  patience=2)

    # Training and validation loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = validate_epoch(model, val_loader, criterion, device)

        # Step the scheduler
        if Schedule:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate

            # or Step the scheduler based on validation loss
            # scheduler.step(val_loss)
            # current_lr = scheduler.get_last_lr()[0]


        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ")
              #f"Learning Rate: {current_lr:.6f}")

        if EralyStopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()  # Save the best model weights
                patience_counter = 0  # Reset counter
            else:
                patience_counter += 1  # Increment counter if no improvement
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    early_stop_epoch = epoch + 1
                    break

    # Restore the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Restored best model weights from epoch with Val Loss: {best_val_loss:.4f}")


    print("✅ Training and validation complete!")


    # Evaluate on the test set
    f1, accuracy = test_model(model, test_loader, device)
    print(f"🔥 Test F1 Score: {f1:.4f}, Test Accuracy: {accuracy:.4f}")

    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    if EralyStopping:
        ax1.plot(range(1, early_stop_epoch + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, early_stop_epoch + 1), val_losses, label="Val Loss")
    else:
        ax1.plot(range(1, epochs + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, epochs + 1), val_losses, label="Val Loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    if EralyStopping:
        ax2.plot(range(1, early_stop_epoch + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, early_stop_epoch + 1), val_accuracies, label="Val Accuracy")
    else:
        ax2.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, epochs + 1), val_accuracies, label="Val Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return model


def train_and_evaluate_with_kfold(train_loader, val_loader, test_loader, epochs=20, lr=0.001, weight_decay=1e-4, k_folds=5, save_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the save directory (default to the directory of the script)
    if save_dir is None:
        # Get the directory of the current script
        save_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Combine train and val datasets for k-fold cross-validation
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    full_train_dataset = ConcatDataset([train_dataset, val_dataset])
    train_val_indices = list(range(len(full_train_dataset)))

    # Define k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Lists to store metrics for each fold
    fold_val_accuracies = []
    fold_val_losses = []
    fold_test_f1_scores = []
    fold_test_accuracies = []
    best_models = []

    # Calculate class weights for the entire train_val dataset
    num_good = 0.66 * len(full_train_dataset)  # 66% good
    num_bad = 0.33 * len(full_train_dataset)   # 33% bad
    weight_good = 1 / num_good
    weight_bad = 1 / num_bad
    class_weights = torch.tensor([weight_good, weight_bad]).to(device)
    class_weights = class_weights / class_weights.sum() * 2  # Normalize

    # K-fold cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        print(f"\nFold {fold+1}/{k_folds}")

        # Create subsets for training and validation
        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_train_dataset, val_idx)

        # Create DataLoaders for this fold
        fold_train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        fold_val_loader = DataLoader(val_subset, batch_size=val_loader.batch_size, shuffle=False)

        # Initialize model, criterion, optimizer, and scheduler
        model = CNN1D_DS().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Training loop for the current fold
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, fold_train_loader, optimizer, criterion, device)
            val_loss, val_acc, misclassified = validate_epoch(model, fold_val_loader, criterion, device)

            scheduler.step(val_loss)
            current_lr = scheduler.optimizer.param_groups[0]['lr']

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print(f"Fold {fold+1} Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                  f"Learning Rate: {current_lr:.6f}")
            print(f"Fold {fold+1} Misclassified validation indices: {misclassified}")

            # Save the model if it has the best validation loss so far
            if val_loss < best_val_loss:
                # Save the model in the specified directory
                checkpoint_path = os.path.join(save_dir, f"best_model_fold_{fold + 1}.ckpt")
                torch.save(model.state_dict(), checkpoint_path)

        # Load the best model for this fold (based on validation loss)
        checkpoint_path = os.path.join(save_dir, f"best_model_fold_{fold + 1}.ckpt")
        model.load_state_dict(torch.load(checkpoint_path))
        best_models.append(model)


        # Evaluate on the test set for this fold
        test_f1, test_acc = test_model(model, test_loader, device)
        print(f"Fold {fold+1} Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")

        # Store metrics for this fold
        fold_val_accuracies.append(val_accuracies[-1])
        fold_val_losses.append(val_losses[-1])
        fold_test_f1_scores.append(test_f1)
        fold_test_accuracies.append(test_acc)

        # Plot metrics for this fold
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Fold {fold+1} Training and Validation Loss")
        ax1.legend()

        ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Fold {fold+1} Training and Validation Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    # Print average metrics across folds
    print("\nCross-Validation Results:")
    print(f"Average Validation Accuracy: {np.mean(fold_val_accuracies):.4f} (±{np.std(fold_val_accuracies):.4f})")
    print(f"Average Validation Loss: {np.mean(fold_val_losses):.4f} (±{np.std(fold_val_losses):.4f})")
    print(f"Average Test F1 Score: {np.mean(fold_test_f1_scores):.4f} (±{np.std(fold_test_f1_scores):.4f})")
    print(f"Average Test Accuracy: {np.mean(fold_test_accuracies):.4f} (±{np.std(fold_test_accuracies):.4f})")

    # Return the model from the fold with the best test accuracy
    best_fold = np.argmax(fold_test_accuracies)
    best_model = best_models[best_fold]
    print(f"\nBest model from fold {best_fold+1} with Test Accuracy: {fold_test_accuracies[best_fold]:.4f}")

    # Save the best model overall in the specified directory
    best_model_path = os.path.join(save_dir, "best_model_overall.ckpt")
    torch.save(best_model.state_dict(), best_model_path)
    print(f"Best model saved at: {best_model_path}")

    return best_model



    # Print average metrics across folds
    print("\nCross-Validation Results:")
    print(f"Average Validation Accuracy: {np.mean(fold_val_accuracies):.4f} (±{np.std(fold_val_accuracies):.4f})")
    print(f"Average Validation Loss: {np.mean(fold_val_losses):.4f} (±{np.std(fold_val_losses):.4f})")
    print(f"Average Test F1 Score: {np.mean(fold_test_f1_scores):.4f} (±{np.std(fold_test_f1_scores):.4f})")
    print(f"Average Test Accuracy: {np.mean(fold_test_accuracies):.4f} (±{np.std(fold_test_accuracies):.4f})")

    # Return the model from the fold with the best test accuracy
    best_fold = np.argmax(fold_test_accuracies)
    best_model = best_models[best_fold]
    print(f"\nBest model from fold {best_fold+1} with Test Accuracy: {fold_test_accuracies[best_fold]:.4f}")

    return best_model

# ------------------------
# 6️⃣ Run Training & Evaluation
# ------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Splitting the dataset
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data"
    dataset = VibrationDataset(data_directory)


    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]

    # Stratified split by both label and operation
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, stratify=stratify_key
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[stratify_key[i] for i in temp_idx]
    )

    # Create Subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Verify split sizes and label distribution
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    print(f"Train good: {sum(dataset.labels[train_idx] == 0)}, Train bad: {sum(dataset.labels[train_idx] == 1)}")
    print(f"Val good: {sum(dataset.labels[val_idx] == 0)}, Val bad: {sum(dataset.labels[val_idx] == 1)}")
    print(f"Test good: {sum(dataset.labels[test_idx] == 0)}, Test bad: {sum(dataset.labels[test_idx] == 1)}")

    # Class ratios
    train_ratio = sum(dataset.labels[train_idx] == 0) / sum(dataset.labels[train_idx] == 1)
    val_ratio = sum(dataset.labels[val_idx] == 0) / sum(dataset.labels[val_idx] == 1)
    test_ratio = sum(dataset.labels[test_idx] == 0) / sum(dataset.labels[test_idx] == 1)
    print(f"Class ratio (good/bad) - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")

    # Operation distribution
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    # Creating DataLoaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_model = train_and_evaluate(train_loader, val_loader, test_loader, EralyStopping=True, Schedule=True, epochs=20, lr=0.001, weight_decay=1e-4)

    # Save the best model
    # Save the trained model

    torch.save(best_model.state_dict(), "../cnn1d_model.ckpt")
    print("✅ Model saved to cnn1d_model.ckpt")
    best_model.to(device)
    best_model.eval()  # Switch to evaluation mode


