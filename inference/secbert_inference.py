#!/usr/bin/env python3
"""
SecBERT Security Severity Inference
===================================
Load the fine-tuned SecBERT model from HuggingFace and predict security severity
from vulnerability descriptions.

Model: https://huggingface.co/huzaifas-sidhpurwala/secbert-redhat-data
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check for required packages
required_packages = ['torch', 'transformers', 'numpy']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("❌ Missing required packages:")
    for pkg in missing_packages:
        print(f"  • {pkg}")
    print("\n🔧 Install them with:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import json

class SecBERTInference:
    """SecBERT inference class for security severity prediction"""
    
    def __init__(self, model_name="huzaifas-sidhpurwala/secbert-redhat-data"):
        """
        Initialize the SecBERT inference model
        
        Args:
            model_name (str): HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._setup_device()
        
        # Define severity classes (based on RedHat VeX dataset)
        self.severity_classes = ['critical', 'important', 'low', 'moderate']
        
        print(f"🔐 SecBERT Security Severity Inference")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
    
    def _setup_device(self):
        """Setup optimal device for Apple Silicon"""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🚀 Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Using Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
        
        return device
    
    def load_model(self):
        """Load the fine-tuned SecBERT model from HuggingFace"""
        print(f"\n📥 Loading model from HuggingFace...")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded successfully!")
            print(f"   • Tokenizer vocabulary size: {len(self.tokenizer)}")
            print(f"   • Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   • Number of classes: {self.model.config.num_labels}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n🔧 Troubleshooting:")
            print("1. Check internet connection")
            print("2. Verify model name is correct")
            print("3. Ensure you have access to the model")
            return False
    
    def predict_severity(self, description, return_all_scores=False):
        """
        Predict security severity from vulnerability description
        
        Args:
            description (str): Vulnerability description text
            return_all_scores (bool): Return scores for all classes
            
        Returns:
            tuple: (predicted_severity, confidence) or (predicted_severity, confidence, all_scores)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            description,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities).item()
            
            # Convert to severity label
            predicted_severity = self.severity_classes[predicted_class_idx]
            
            if return_all_scores:
                all_scores = {
                    self.severity_classes[i]: probabilities[0][i].item()
                    for i in range(len(self.severity_classes))
                }
                return predicted_severity, confidence, all_scores
            else:
                return predicted_severity, confidence
    
    def predict_batch(self, descriptions):
        """
        Predict severity for multiple descriptions
        
        Args:
            descriptions (list): List of vulnerability descriptions
            
        Returns:
            list: List of (description, predicted_severity, confidence) tuples
        """
        results = []
        
        print(f"🔄 Processing {len(descriptions)} descriptions...")
        
        for i, desc in enumerate(descriptions, 1):
            try:
                severity, confidence = self.predict_severity(desc)
                results.append((desc, severity, confidence))
                print(f"   {i}/{len(descriptions)}: {severity.upper()} ({confidence:.3f})")
            except Exception as e:
                print(f"   {i}/{len(descriptions)}: ERROR - {e}")
                results.append((desc, "ERROR", 0.0))
        
        return results
    
    def interactive_mode(self):
        """Run interactive prediction mode"""
        print(f"\n{'='*60}")
        print("🧪 INTERACTIVE SECBERT INFERENCE")
        print("="*60)
        print("Enter vulnerability descriptions to get severity predictions.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                # Get user input
                description = input("📝 Enter description: ").strip()
                
                if description.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not description:
                    print("⚠️  Please enter a description.")
                    continue
                
                # Get prediction
                severity, confidence, all_scores = self.predict_severity(
                    description, return_all_scores=True
                )
                
                # Display results
                print(f"\n🎯 Prediction Results:")
                print(f"   Predicted Severity: {severity.upper()}")
                print(f"   Confidence: {confidence:.3f} ({confidence:.1%})")
                
                print(f"\n📊 All Scores:")
                for sev, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(score * 20)
                    print(f"   {sev:>9}: {score:.3f} {bar}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description="SecBERT Security Severity Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python secbert_inference.py

  # Single prediction
  python secbert_inference.py --text "Buffer overflow in HTTP parser"

  # Batch prediction from file
  python secbert_inference.py --file descriptions.txt

  # JSON output
  python secbert_inference.py --text "XSS vulnerability" --json
        """
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Single vulnerability description to classify'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='File containing vulnerability descriptions (one per line)'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default="huzaifas-sidhpurwala/secbert-redhat-data",
        help='HuggingFace model name (default: huzaifas-sidhpurwala/secbert-redhat-data)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = SecBERTInference(model_name=args.model)
    
    # Load model
    if not inference.load_model():
        return 1
    
    # Handle different modes
    if args.text:
        # Single text prediction
        try:
            severity, confidence, all_scores = inference.predict_severity(
                args.text, return_all_scores=True
            )
            
            if args.json:
                result = {
                    "input": args.text,
                    "predicted_severity": severity,
                    "confidence": confidence,
                    "all_scores": all_scores
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"\n🎯 Prediction:")
                print(f"Input: {args.text}")
                print(f"Predicted Severity: {severity.upper()}")
                print(f"Confidence: {confidence:.3f} ({confidence:.1%})")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    elif args.file:
        # Batch prediction from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                descriptions = [line.strip() for line in f if line.strip()]
            
            results = inference.predict_batch(descriptions)
            
            if args.json:
                json_results = [
                    {
                        "input": desc,
                        "predicted_severity": severity,
                        "confidence": confidence
                    }
                    for desc, severity, confidence in results
                ]
                print(json.dumps(json_results, indent=2))
            else:
                print(f"\n📊 Batch Results:")
                for i, (desc, severity, confidence) in enumerate(results, 1):
                    print(f"{i:3d}. {severity.upper():>9} ({confidence:.3f}) - {desc[:60]}...")
                    
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            return 1
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    else:
        # Interactive mode
        inference.interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())