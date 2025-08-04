#!/usr/bin/env python3
"""
Security Severity Classification Model
=====================================
Fine-tunes a transformer model to classify security issue severity
from RedHat VeX dataset descriptions and metadata.

Target classes: critical, important, moderate, low
Expected accuracy: >90%
"""

import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class SecurityDataset(Dataset):
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

class SecuritySeverityClassifier:
    """Main classifier class for security severity prediction"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.trainer = None
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the RedHat VeX dataset"""
        print("Loading RedHat security VeX dataset...")
        
        # Load dataset from HuggingFace
        dataset = load_dataset("huzaifas-sidhpurwala/RedHat-security-VeX")
        df = dataset['train'].to_pandas()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean and prepare the data
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
        
        print("Severity distribution:")
        print(df['severity_clean'].value_counts())
        
        return df
    
    def prepare_model_and_tokenizer(self):
        """Initialize tokenizer and model"""
        print(f"Loading {self.model_name} tokenizer and model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_datasets(self, df, test_size=0.2, val_size=0.1):
        """Split data and create datasets"""
        print("Preparing train/validation/test splits...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(df['severity_clean'])
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            df['combined_text'], y_encoded, 
            test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        print(f"Train size: {len(X_train)}")
        print(f"Validation size: {len(X_val)}")
        print(f"Test size: {len(X_test)}")
        
        # Create datasets
        train_dataset = SecurityDataset(X_train, y_train, self.tokenizer)
        val_dataset = SecurityDataset(X_val, y_val, self.tokenizer)
        test_dataset = SecurityDataset(X_test, y_test, self.tokenizer)
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(labels, predictions)}
    
    def train_model(self, train_dataset, val_dataset, output_dir='./security_model'):
        """Train the classification model"""
        print("Starting model training...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the model and tokenizer
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save the label encoder for consistent predictions
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Training completed! Model saved to {output_dir}")
        print(f"Label encoder saved for consistent predictions")
        return train_result
    
    def evaluate_model(self, test_dataset, show_plots=True):
        """Evaluate the trained model"""
        print("Evaluating model on test set...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if show_plots:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names)
            plt.title('Confusion Matrix - Security Severity Classification')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return accuracy, report, cm
    
    def predict_severity(self, text):
        """Predict severity for new text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained or loaded!")
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors='pt'
        )
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
        
        severity = self.label_encoder.inverse_transform([predicted_class])[0]
        return severity, confidence

def main():
    """Main training and evaluation pipeline"""
    
    # Initialize classifier
    classifier = SecuritySeverityClassifier(
        model_name='distilbert-base-uncased',  # Can also try 'roberta-base' for potentially better performance
        num_labels=4
    )
    
    try:
        # Load and preprocess data
        df = classifier.load_and_preprocess_data()
        
        # Prepare model and tokenizer
        classifier.prepare_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(df)
        
        # Train model
        train_result = classifier.train_model(train_dataset, val_dataset)
        
        # Evaluate model
        accuracy, report, cm = classifier.evaluate_model(test_dataset)
        
        # Test predictions on sample data
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS:")
        print("="*50)
        
        sample_texts = [
            "Buffer overflow vulnerability in network daemon allows remote code execution",
            "Information disclosure through log files containing sensitive data",
            "Cross-site scripting vulnerability in web application",
            "Denial of service through resource exhaustion in HTTP parser"
        ]
        
        for text in sample_texts:
            severity, confidence = classifier.predict_severity(text)
            print(f"Text: {text[:80]}...")
            print(f"Predicted Severity: {severity} (confidence: {confidence:.3f})")
            print("-" * 50)
        
        print(f"\nFinal Test Accuracy: {accuracy:.4f}")
        if accuracy >= 0.90:
            print("✅ SUCCESS: Model achieved target accuracy of 90%+")
        else:
            print("⚠️  WARNING: Model accuracy below 90% target")
            print("Consider: longer training, different model, or data augmentation")
            
        print("\nTo test on the full dataset, run: python test_security_model.py")
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Please check your internet connection and try again.")
        return None
    
    return classifier

if __name__ == "__main__":
    classifier = main()