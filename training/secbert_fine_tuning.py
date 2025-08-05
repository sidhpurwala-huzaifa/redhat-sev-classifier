#!/usr/bin/env python3
"""
SecBERT Fine-tuning for Security Severity Classification
======================================================
Fine-tunes SecBERT model using RedHat VeX dataset to predict vulnerability severity
from descriptions. Optimized for Apple Silicon.

Usage: python secbert_fine_tuning.py
"""

import os
import sys

# Check for required packages with helpful error messages
required_packages = {
    'pandas': 'pandas>=2.0.0',
    'numpy': 'numpy>=1.24.0', 
    'torch': 'torch>=2.0.0',
    'transformers': 'transformers>=4.30.0',
    'sklearn': 'scikit-learn>=1.3.0',
    'datasets': 'datasets>=2.10.0',
    'matplotlib': 'matplotlib>=3.7.0',
    'seaborn': 'seaborn>=0.12.0'
}

missing_packages = []
for package, version_info in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(version_info)

if missing_packages:
    print("❌ Missing required packages:")
    for pkg in missing_packages:
        print(f"  • {pkg}")
    print("\n🔧 Install them with:")
    print(f"pip install {' '.join(missing_packages)}")
    print("\n💡 Or activate your virtual environment first:")
    print("source venv/bin/activate  # or source .venv/bin/activate")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Now import everything
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
        # Handle both pandas Series and numpy arrays safely
        try:
            text = str(self.texts.iloc[idx])
            label = self.labels.iloc[idx]
        except (AttributeError, TypeError):
            text = str(self.texts[idx])
            label = self.labels[idx]
        
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

class SecBERTSeverityClassifier:
    """SecBERT-based classifier for security severity prediction"""
    
    def __init__(self, model_name='jackaduma/SecBERT'):
        self.model_name = model_name
        self.num_labels = 4  # critical, important, moderate, low
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.trainer = None
        
        # Set up Apple Silicon optimized device
        self.device = self._setup_device()
        print(f"Using device: {self.device}")
    
    def _setup_device(self):
        """Setup optimal device for Apple Silicon"""
        # Disable logging integrations
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled" 
        os.environ["TENSORBOARD_DISABLED"] = "true"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🚀 Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Using Apple Silicon MPS")
        else:
            device = torch.device('mps')
            print("💻 Using MPS")
            # Optimize CPU for Apple Silicon
            torch.set_num_threads(8)
        
        return device
    
    def load_and_preprocess_data(self):
        """Load and preprocess the RedHat VeX dataset"""
        print("📥 Loading RedHat Security VeX dataset from HuggingFace...")
        
        # Load dataset from HuggingFace
        dataset = load_dataset("huzaifas-sidhpurwala/RedHat-security-VeX")
        df = dataset['train'].to_pandas()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Clean and prepare the data - focus on required fields
        df = df.dropna(subset=['severity', 'description'])
        
        # Combine summary and description as requested
        # Use description as primary text, add summary if available
        df['text_input'] = df.apply(lambda row: 
            str(row['summary'] or '') + ' ' + str(row['description'] or ''), axis=1
        ).str.strip()
        
        # Remove empty text inputs
        df = df[df['text_input'].str.len() > 0]
        
        # Clean severity labels - map to standard format
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
        
        print("\n📊 Severity distribution:")
        severity_counts = df['severity_clean'].value_counts()
        print(severity_counts)
        print(f"\nTotal samples: {len(df)}")
        
        return df
    
    def prepare_model_and_tokenizer(self):
        """Initialize SecBERT tokenizer and model"""
        print(f"🤖 Loading {self.model_name} tokenizer and model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
        except Exception as e:
            print(f"❌ Error loading SecBERT: {e}")
            print("🔄 Falling back to BERT...")
            self.model_name = 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"✅ Model loaded and moved to {self.device}")
    
    def prepare_datasets(self, df, test_size=0.2, val_size=0.1):
        """Split data into train/validation/test sets"""
        print("📊 Preparing train/validation/test splits...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(df['severity_clean'])
        print(f"Label classes: {self.label_encoder.classes_}")
        
        # Split data - stratified to maintain class distribution
        X_temp, X_test, y_temp, y_test = train_test_split(
            df['text_input'].reset_index(drop=True), 
            y_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, 
            y_temp,
            test_size=val_size/(1-test_size), 
            random_state=42, 
            stratify=y_temp
        )
        
        # Ensure proper data types
        X_train = pd.Series(X_train).reset_index(drop=True)
        X_val = pd.Series(X_val).reset_index(drop=True)
        X_test = pd.Series(X_test).reset_index(drop=True)
        
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)
        
        print(f"Train size: {len(X_train)} ({len(X_train)/len(df):.1%})")
        print(f"Validation size: {len(X_val)} ({len(X_val)/len(df):.1%})")
        print(f"Test size: {len(X_test)} ({len(X_test)/len(df):.1%})")
        
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
    
    def train_model(self, train_dataset, val_dataset, output_dir='./secbert_model'):
        """Train the SecBERT classification model"""
        print("🎯 Starting SecBERT fine-tuning...")
        
        # Force CPU mode for maximum compatibility with AdamW
        print("🔧 Using CPU mode for AdamW compatibility...")
        self.device = torch.device('mps')
        self.model = self.model.to(self.device)
        
        # CPU optimized settings
        train_batch_size = 4  # Smaller batch for stability
        eval_batch_size = 8
        dataloader_num_workers = 0  # Avoid multiprocessing issues
        
        print(f"Batch sizes - Train: {train_batch_size}, Eval: {eval_batch_size}")
        print("💡 Note: Using CPU mode avoids AdamW/MPS compatibility issues")
        
        # Create training arguments with stable optimizer settings
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            warmup_steps=100,  # Reduced warmup steps
            weight_decay=0.01,
            learning_rate=2e-5,  # Explicit learning rate
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,  # More frequent logging
            evaluation_strategy="steps",
            eval_steps=200,  # More frequent evaluation
            save_strategy="steps", 
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            report_to=[],  # No online logging
            remove_unused_columns=False,
            fp16=False,  # Disable mixed precision
            bf16=False,  # Disable bfloat16
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=False,  # Disable for CPU
            optim='adamw_torch',  # Use PyTorch AdamW
            seed=42,  # Set seed for reproducibility
            data_seed=42,
            no_cuda=True,  # Force CPU usage
        )
        
        # Put model in training mode before creating trainer
        self.model.train()
        
        # Create trainer with error handling
        try:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            print(f"🚀 Training {self.model_name} on {self.device}...")
            print("⏱️  Expected training time: 30-45 minutes on Apple Silicon CPU")
            
            # Train the model with error handling
            train_result = self.trainer.train()
            
            # Save everything
            print("💾 Saving trained model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save label encoder
            with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"✅ Training completed! Model saved to {output_dir}")
            return train_result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Training error: {error_msg}")
            
            if "'AdamW' object has no attribute 'train'" in error_msg:
                print("\n🔧 AdamW Compatibility Issue Detected")
                print("💡 This is a known issue with certain transformers versions")
                print("🔄 Trying fallback training configuration...")
                
                # Retry with even more conservative settings
                return self._train_with_fallback(train_dataset, val_dataset, output_dir)
            else:
                raise e
    
    def _train_with_fallback(self, train_dataset, val_dataset, output_dir):
        """Fallback training method for AdamW issues"""
        print("🔄 Attempting fallback training with conservative settings...")
        
        # Even more conservative settings
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Reduced epochs
            per_device_train_batch_size=2,  # Very small batch
            per_device_eval_batch_size=4,
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=5e-5,  # Higher learning rate for faster training
            logging_steps=25,
            evaluation_strategy="epoch",  # Evaluate per epoch instead
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy", 
            greater_is_better=True,
            save_total_limit=1,
            report_to=[],
            remove_unused_columns=False,
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            optim='sgd',  # Use SGD instead of AdamW
            seed=42,
            data_seed=42,
            no_cuda=True,
        )
        
        # Create new trainer with fallback settings
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        print("🚀 Starting fallback training with SGD optimizer...")
        train_result = self.trainer.train()
        
        # Save everything
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ Fallback training completed! Model saved to {output_dir}")
        return train_result
    
    def evaluate_model(self, test_dataset, show_plots=True):
        """Evaluate the trained model"""
        print("📈 Evaluating model on test set...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        
        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if show_plots:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names)
            plt.title('SecBERT Security Severity Classification - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('secbert_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Confusion matrix saved as 'secbert_confusion_matrix.png'")
        
        return accuracy, report, cm
    
    def predict_severity(self, text):
        """Predict severity for new text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained or loaded!")
        
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
        
        severity = self.label_encoder.inverse_transform([predicted_class])[0]
        return severity, confidence

def main():
    """Main fine-tuning pipeline"""
    print("🔐 SecBERT Fine-tuning for Security Severity Classification")
    print("=" * 60)
    
    # Initialize classifier
    classifier = SecBERTSeverityClassifier()
    
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
        
        # Test sample predictions
        print("\n" + "="*60)
        print("🧪 SAMPLE PREDICTIONS:")
        print("="*60)
        
        sample_texts = [
            "Buffer overflow vulnerability allows remote code execution with system privileges",
            "Cross-site scripting vulnerability in web interface allows session hijacking", 
            "Information disclosure through verbose error messages in logs",
            "Denial of service through resource exhaustion in HTTP parser"
        ]
        
        for i, text in enumerate(sample_texts, 1):
            severity, confidence = classifier.predict_severity(text)
            print(f"\n{i}. Text: {text}")
            print(f"   Predicted Severity: {severity.upper()} (confidence: {confidence:.3f})")
        
        # Final results
        print(f"\n{'='*60}")
        print("🏆 FINAL RESULTS")
        print("="*60)
        print(f"Model: {classifier.model_name}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"Target Classes: {list(classifier.label_encoder.classes_)}")
        
        if accuracy >= 0.90:
            print("✅ SUCCESS: Model achieved target accuracy of 90%+")
        else:
            print("⚠️  Model accuracy below 90% target")
            print("💡 Consider longer training or hyperparameter tuning")
        
        print("\n📁 Output files:")
        print("  • ./secbert_model/ - Trained model and tokenizer")
        print("  • secbert_confusion_matrix.png - Performance visualization")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection for dataset download")
        print("2. Ensure sufficient disk space (~1GB)")
        print("3. Install requirements: pip install transformers datasets scikit-learn matplotlib seaborn")
        if "MPS" in str(e):
            print("4. Try: export PYTORCH_ENABLE_MPS_FALLBACK=1")
        
        return None
    
    return classifier

if __name__ == "__main__":
    classifier = main()
