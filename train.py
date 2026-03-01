import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np

from dataset import get_mnist_dataloaders
from model2 import HLGAN

def train_model(
    epochs=15,
    batch_size=64,
    learning_rate=1e-3,
    device=None
):
    # Select device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print("="*60 + "\n")

    # Load dataset
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    # Initialize model
    model = HLGAN().to(device)
    print(f"Model loaded: HLGAN")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*60 + "\n")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_f1 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass (optimization)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        metrics = evaluate_model(model, test_loader, device)

        # Save best model based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_state = model.state_dict().copy()
            print("[OK] New best model saved (F1 Score improved)")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n{'='*60}")
        print("FINAL RESULTS (Best Model)")
        print(f"{'='*60}")
        evaluate_model(model, test_loader, device)

    return model

def evaluate_model(model, test_loader, device):
    """
    Comprehensive evaluation with multiple metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # ---- Compute Metrics ----
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # ---- Print Results ----
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50 + "\n")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    print()

    # ---- Classification Report (per-class metrics) ----
    print("Classification Report (Per-Class Metrics):")
    print(classification_report(all_labels, all_predictions,
                                target_names=[str(i) for i in range(10)]))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    train_model(
        epochs=15,
        batch_size=64,
        learning_rate=1e-3
    )
