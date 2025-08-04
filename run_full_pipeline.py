#!/usr/bin/env python3
"""
Complete Security Severity Classification Pipeline
=================================================
Runs the full training and evaluation pipeline on the HuggingFace dataset
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print("="*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'scikit-learn', 
        'pandas', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing requirements...")
        if not run_command("pip install -r requirements_security.txt", "Installing dependencies"):
            return False
    
    return True

def main():
    """Run the complete pipeline"""
    print("Security Severity Classification - Full Pipeline")
    print("=" * 60)
    print("This will:")
    print("1. Check dependencies")
    print("2. Train model on HuggingFace RedHat VeX dataset")
    print("3. Test model on real dataset (not static examples)")
    print("4. Show comprehensive evaluation results")
    
    # Check if user wants to continue
    response = input("\nContinue? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Step 2: Train the model
    print(f"\n{'='*60}")
    print("TRAINING MODEL ON HUGGINGFACE DATASET")
    print("="*60)
    print("This will download ~170MB of security vulnerability data")
    print("and train a DistilBERT model for severity classification...")
    
    if not run_command("python security_severity_classifier.py", "Training model"):
        print("❌ Training failed")
        return
    
    # Step 3: Test on real dataset
    print(f"\n{'='*60}")
    print("TESTING ON REAL DATASET TEST SPLIT")
    print("="*60)
    print("This will evaluate the trained model on ~9,800 real")
    print("security vulnerability descriptions from the test set...")
    
    if not run_command("python test_security_model.py", "Testing on dataset"):
        print("❌ Testing failed")
        return
    
    # Success message
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ Model trained on HuggingFace RedHat VeX dataset")
    print("✅ Evaluated on real test data (not static examples)")
    print("✅ Results saved and displayed")
    print("\nFiles generated:")
    print("- ./security_model/ (trained model)")
    print("- confusion_matrix.png (training evaluation)")
    print("- dataset_test_confusion_matrix.png (test evaluation)")
    
    print("\nTo use the model for predictions:")
    print("```python")
    print("from security_severity_classifier import SecuritySeverityClassifier")
    print("classifier = SecuritySeverityClassifier()")
    print("# Load your trained model and make predictions")
    print("```")

if __name__ == "__main__":
    main()