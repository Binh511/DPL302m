import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tqdm import tqdm
import sys
import io
import logging

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for easy parameter management"""
    MODEL_NAME = "albert-base-v2"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    EPOCHS = 6       
    LEARNING_RATE = 2e-5
    NUM_LABELS = 28
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GoEmotionsDataset(Dataset):
    """Dataset class for GoEmotions"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def set_seed(seed=42):
    """Set seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    logger.info("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified")
    
    def filter_single_label(texts, labels):
        """Keep only samples with exactly one label"""
        filtered_texts = []
        filtered_labels = []
        for text, lbl in zip(texts, labels):
            if len(lbl) == 1:  # Single label only
                filtered_texts.append(text)
                filtered_labels.append(lbl[0])
        return filtered_texts, filtered_labels
    
    logger.info("Filtering multi-label samples...")
    train_texts, train_labels = filter_single_label(
        dataset['train']['text'], 
        dataset['train']['labels']
    )
    
    val_texts, val_labels = filter_single_label(
        dataset['validation']['text'], 
        dataset['validation']['labels']
    )
    
    test_texts, test_labels = filter_single_label(
        dataset['test']['text'], 
        dataset['test']['labels']
    )
    
    # Print statistics
    logger.info("\nDataset statistics (after filtering multi-label):")
    logger.info(f"  Train: {len(dataset['train']['text']):,} -> {len(train_texts):,} samples")
    logger.info(f"  Val:   {len(dataset['validation']['text']):,} -> {len(val_texts):,} samples")
    logger.info(f"  Test:  {len(dataset['test']['text']):,} -> {len(test_texts):,} samples")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Training loop for one epoch
    
    Returns:
        avg_loss, accuracy, f1_score
    """
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, device):
    """
    Evaluation loop
    
    Returns:
        avg_loss, accuracy, f1_score
    """
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1


def main():
    """Main training pipeline"""
    # Set seed for reproducibility
    set_seed(42)
    
    config = Config()
    
    # Print configuration
    logger.info("="*70)
    logger.info("Training ALBERT-base-v2 for Single-Label Classification")
    logger.info("="*70)
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info("="*70)
    
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data()
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Train samples: {len(train_texts):,}")
    logger.info(f"  Val samples: {len(val_texts):,}")
    logger.info(f"  Test samples: {len(test_texts):,}")
    
    # Initialize tokenizer and model
    logger.info(f"\nLoading ALBERT-base-v2 model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS
    )
    model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    train_dataset = GoEmotionsDataset(
        train_texts, train_labels, tokenizer, config.MAX_LENGTH
    )
    val_dataset = GoEmotionsDataset(
        val_texts, val_labels, tokenizer, config.MAX_LENGTH
    )
    test_dataset = GoEmotionsDataset(
        test_texts, test_labels, tokenizer, config.MAX_LENGTH
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE,
        num_workers=2
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info(f"\nStarting training...")
    logger.info(f"Total training steps: {total_steps:,}")
    
    best_val_f1 = 0
    training_history = []
    patience = 2
    no_improve_epochs = 0
    
    for epoch in range(config.EPOCHS):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")
        logger.info(f"{'='*70}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE
        )
        logger.info(
            f"\nTrain Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}"
        )
        
        # Validate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, config.DEVICE)
        logger.info(
            f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
        )
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })
        
        # Calculate improvement
        improvement = val_f1 - best_val_f1
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logger.info(f"\n✓ New best model! Saving (F1: {val_f1:.4f})")
            logger.info(f"   Improvement: +{improvement:.4f}")
            model.save_pretrained('./best_albert_single_label')
            tokenizer.save_pretrained('./best_albert_single_label')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logger.info(f"\nNo improvement for {no_improve_epochs} epoch(s)")
            
            # Early stopping
            if no_improve_epochs >= patience:
                logger.info(f"\nEarly stopping after {epoch + 1} epochs.")
                break
    
    # Load best model for testing
    logger.info(f"\n{'='*70}")
    logger.info("Loading best model for final evaluation...")
    logger.info(f"{'='*70}")
    model = AutoModelForSequenceClassification.from_pretrained('./best_albert_single_label')
    model.to(config.DEVICE)
    
    # Test on best model
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, config.DEVICE)
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    
    # Print training history summary
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING HISTORY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Epoch':<8} {'Train Loss':<12} {'Train F1':<10} {'Val Loss':<12} {'Val F1':<10}")
    logger.info("-"*70)
    for h in training_history:
        logger.info(
            f"{h['epoch']:<8} {h['train_loss']:<12.4f} {h['train_f1']:<10.4f} "
            f"{h['val_loss']:<12.4f} {h['val_f1']:<10.4f}"
        )
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING COMPLETED!")
    logger.info(f"{'='*70}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"Model saved at: ./best_albert_single_label")
    logger.info("="*70)

if __name__ == "__main__":

    main()
