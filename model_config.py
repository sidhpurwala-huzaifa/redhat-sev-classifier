#!/usr/bin/env python3
"""
Configuration file for Security Severity Classification
=====================================================
Contains model configurations and hyperparameters
"""

# Model configurations
MODEL_CONFIGS = {
    'distilbert': {
        'name': 'distilbert-base-uncased',
        'description': 'Lightweight BERT model, fast training',
        'max_length': 512,
        'batch_size_train': 16,
        'batch_size_eval': 32,
        'expected_accuracy': 0.92
    },
    'roberta': {
        'name': 'roberta-base',
        'description': 'RoBERTa model, potentially better performance',
        'max_length': 512,
        'batch_size_train': 12,
        'batch_size_eval': 24,
        'expected_accuracy': 0.94
    },
    'bert': {
        'name': 'bert-base-uncased',
        'description': 'Original BERT model',
        'max_length': 512,
        'batch_size_train': 12,
        'batch_size_eval': 24,
        'expected_accuracy': 0.93
    },
    'security_bert': {
        'name': 'jackaduma/SecBERT',
        'description': 'Security-domain fine-tuned BERT',
        'max_length': 512,
        'batch_size_train': 8,
        'batch_size_eval': 16,
        'expected_accuracy': 0.95
    }
}

# Training hyperparameters
TRAINING_CONFIG = {
    'num_epochs': 4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'learning_rate': 2e-5,
    'eval_steps': 500,
    'save_steps': 500,
    'early_stopping_patience': 3,
    'test_size': 0.2,
    'val_size': 0.1
}

# Severity mappings
SEVERITY_MAPPING = {
    'Critical': 'critical',
    'Important': 'important', 
    'Moderate': 'moderate',
    'Low': 'low',
    'critical': 'critical',
    'important': 'important',
    'moderate': 'moderate',
    'low': 'low',
    'CRITICAL': 'critical',
    'IMPORTANT': 'important',
    'MODERATE': 'moderate',
    'LOW': 'low'
}

# Feature engineering options
TEXT_FEATURES = {
    'basic': ['description'],
    'enhanced': ['description', 'summary'],
    'comprehensive': ['description', 'summary', 'cwe', 'affected_component'],
    'full': ['description', 'summary', 'cwe', 'affected_component', 'statement']
}

def get_model_config(model_type='distilbert'):
    """Get configuration for a specific model type"""
    if model_type not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Model type '{model_type}' not found. Available: {available}")
    
    return MODEL_CONFIGS[model_type]

def print_available_models():
    """Print information about available models"""
    print("Available Model Configurations:")
    print("=" * 50)
    for key, config in MODEL_CONFIGS.items():
        print(f"\n{key.upper()}:")
        print(f"  Model: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Expected Accuracy: {config['expected_accuracy']:.1%}")

if __name__ == "__main__":
    print_available_models()