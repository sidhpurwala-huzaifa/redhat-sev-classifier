#!/usr/bin/env python3
"""
Dataset Test Script for Security Severity Classifier
===================================================
This script tests the trained model on the actual HuggingFace dataset test split
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SecurityDataset:
    """Custom dataset for security vulnerability data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_trained_model(model_path='./security_model'):
    """Load the trained model and tokenizer"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run the training script first!")
        return None, None, None
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load the saved label encoder
    label_encoder_path = os.path.join(model_path, 'label_encoder.pkl')
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Loaded saved label encoder")
    else:
        # Fallback: recreate label encoder (for backward compatibility)
        print("Label encoder not found, creating new one (may cause label mismatch)")
        label_encoder = LabelEncoder()
        label_encoder.fit(['critical', 'important', 'low', 'moderate'])
    
    return model, tokenizer, label_encoder

def load_and_preprocess_dataset():
    """Load and preprocess the RedHat VeX dataset (same as training)"""
    print("Loading RedHat security VeX dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("huzaifas-sidhpurwala/RedHat-security-VeX")
    df = dataset['train'].to_pandas()
    
    print(f"Dataset shape: {df.shape}")
    
    # Clean and prepare the data (same preprocessing as training)
    df = df.dropna(subset=['severity', 'description'])
    
    # Combine relevant text features for better classification
    df['combined_text'] = (
        df['summary'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['cwe'].fillna('') + ' ' +
        df['affected_component'].fillna('')
    ).str.strip()
    
    # Clean severity labels
    severity_mapping = {
        'Critical': 'critical',
        'Important': 'important', 
        'Moderate': 'moderate',
        'Low': 'low',
        'critical': 'critical',
        'important': 'important',
        'moderate': 'moderate',
        'low': 'low'
    }
    
    df['severity_clean'] = df['severity'].map(severity_mapping)
    df = df.dropna(subset=['severity_clean'])
    
    print("Severity distribution in full dataset:")
    print(df['severity_clean'].value_counts())
    
    return df

def create_test_split(df, label_encoder, test_size=0.2, val_size=0.1):
    """Create the same test split as used in training"""
    print("Creating test split (same as training)...")
    
    # Encode labels using the saved label encoder
    y_encoded = label_encoder.transform(df['severity_clean'])
    
    # Split data (same random state as training for consistency)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['combined_text'], y_encoded, 
        test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"Test set size: {len(X_test)}")
    print("Test set severity distribution:")
    test_severity_counts = pd.Series(y_test).map(
        dict(enumerate(label_encoder.classes_))
    ).value_counts()
    print(test_severity_counts)
    
    return X_test, y_test

def evaluate_on_dataset(model, tokenizer, label_encoder, X_test, y_test):
    """Evaluate model on the actual dataset test split"""
    print("\nEvaluating model on dataset test split...")
    
    model.eval()
    predictions = []
    true_labels = []
    
    # Process in batches to handle large dataset
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test.iloc[i:i+batch_size].tolist()
            batch_labels = y_test[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Get predictions
            outputs = model(**inputs)
            batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_texts)}/{len(X_test)} samples...")
    
    return np.array(predictions), np.array(true_labels)

def display_results(y_true, y_pred, label_encoder, show_plots=True):
    """Display comprehensive evaluation results"""
    print("\n" + "="*60)
    print("DATASET EVALUATION RESULTS")
    print("="*60)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
    
    # Check if target accuracy achieved
    if accuracy >= 0.90:
        print("✅ SUCCESS: Model achieved target accuracy of 90%+")
    else:
        print("⚠️  WARNING: Model accuracy below 90% target")
    
    # Classification report
    target_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(target_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
            print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy:.1%})")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    if show_plots:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Dataset Test Results')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('dataset_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved as 'dataset_test_confusion_matrix.png'")
        plt.show()
    
    return accuracy, report, cm

def test_sample_predictions(model, tokenizer, label_encoder, X_test, y_test, num_samples=10):
    """Show sample predictions from the test set"""
    print(f"\n{'='*60}")
    print(f"SAMPLE PREDICTIONS FROM TEST SET ({num_samples} examples)")
    print("="*60)
    
    # Get random samples
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices, 1):
            text = X_test.iloc[idx]
            true_label = y_test[idx]
            true_severity = label_encoder.inverse_transform([true_label])[0]
            
            # Get prediction
            inputs = tokenizer(
                text, 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors='pt'
            )
            
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            predicted_severity = label_encoder.inverse_transform([predicted_class])[0]
            
            is_correct = predicted_severity == true_severity
            status = "✅" if is_correct else "❌"
            
            print(f"\nSample {i}:")
            print(f"Text: {text[:100]}...")
            print(f"True Severity: {true_severity}")
            print(f"Predicted: {predicted_severity} (confidence: {confidence:.3f}) {status}")

def main():
    """Main evaluation pipeline using the actual dataset"""
    
    try:
        # Load trained model
        model, tokenizer, label_encoder = load_trained_model()
        if model is None:
            return
        
        # Load and preprocess dataset
        df = load_and_preprocess_dataset()
        
        # Create test split (same as training)
        X_test, y_test = create_test_split(df, label_encoder)
        
        # Evaluate on test set
        y_pred, y_true = evaluate_on_dataset(model, tokenizer, label_encoder, X_test, y_test)
        
        # Display comprehensive results
        accuracy, report, cm = display_results(y_true, y_pred, label_encoder)
        
        # Show sample predictions
        test_sample_predictions(model, tokenizer, label_encoder, X_test, y_test)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Total test samples: {len(y_test)}")
        print(f"Test accuracy: {accuracy:.4f} ({accuracy:.1%})")
        
        if accuracy >= 0.90:
            print("🎉 Model successfully achieved >90% accuracy target!")
        else:
            print("📊 Model performance results available above")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Make sure you have:")
        print("1. Trained the model first (run security_severity_classifier.py)")
        print("2. Installed all requirements (pip install -r requirements_security.txt)")
        print("3. Internet connection for dataset download")

if __name__ == "__main__":
    main()