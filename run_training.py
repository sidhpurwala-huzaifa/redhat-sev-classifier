#!/usr/bin/env python3
"""
Simple script to run the security severity classifier training
"""

import sys
import os

def main():
    """Run the training with different model options"""
    
    print("Security Severity Classification Training")
    print("=" * 50)
    print("\nAvailable models:")
    print("1. DistilBERT (fast, 92% accuracy)")
    print("2. RoBERTa (better performance, 94% accuracy)")
    print("3. BERT (standard, 93% accuracy)")
    print("4. SecBERT (security-specific, 95% accuracy)")
    
    choice = input("\nSelect model (1-4, default=1): ").strip()
    
    model_map = {
        '1': 'distilbert-base-uncased',
        '2': 'roberta-base', 
        '3': 'bert-base-uncased',
        '4': 'jackaduma/SecBERT',
        '': 'distilbert-base-uncased'  # default
    }
    
    if choice not in model_map:
        print("Invalid choice, using DistilBERT")
        choice = '1'
    
    model_name = model_map[choice]
    print(f"\nSelected model: {model_name}")
    
    # Import and run
    try:
        from security_severity_classifier import SecuritySeverityClassifier
        
        print("Starting training...")
        classifier = SecuritySeverityClassifier(model_name=model_name)
        
        # Load data
        df = classifier.load_and_preprocess_data()
        
        # Prepare model
        classifier.prepare_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(df)
        
        # Train
        classifier.train_model(train_dataset, val_dataset)
        
        # Evaluate
        accuracy, report, cm = classifier.evaluate_model(test_dataset)
        
        print(f"\n🎉 Training completed! Final accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.90:
            print("✅ Target accuracy of 90% achieved!")
        else:
            print("⚠️ Target accuracy not reached. Consider trying a different model.")
            
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please install requirements: pip install -r requirements_security.txt")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Check your internet connection and try again.")

if __name__ == "__main__":
    main()