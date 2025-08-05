# SecBERT Security Severity Inference

This package provides inference capabilities for the fine-tuned SecBERT model hosted at [huzaifas-sidhpurwala/secbert-redhat-data](https://huggingface.co/huzaifas-sidhpurwala/secbert-redhat-data).

## 🚀 Quick Start

### Single Prediction
```bash
python secbert_inference.py --text "Buffer overflow vulnerability allows remote code execution"
```

### Interactive Mode
```bash
python secbert_inference.py
```

### Batch Processing
```bash
python secbert_inference.py --file test_descriptions.txt
```

### JSON Output
```bash
python secbert_inference.py --text "XSS vulnerability in web app" --json
```

## 📋 Usage Examples

### 1. Command Line Interface

```bash
# Single prediction with detailed output
python secbert_inference.py --text "SQL injection in login form allows database access"

# Output:
# 🎯 Prediction:
# Input: SQL injection in login form allows database access
# Predicted Severity: IMPORTANT
# Confidence: 0.892 (89.2%)
```

### 2. Batch Processing

Create a file `descriptions.txt`:
```
Buffer overflow in network daemon
Cross-site scripting in web form
Information leak in error messages
```

Run batch inference:
```bash
python secbert_inference.py --file descriptions.txt
```

### 3. Python API

```python
from secbert_inference import SecBERTInference

# Initialize
inference = SecBERTInference()
inference.load_model()

# Single prediction
severity, confidence = inference.predict_severity(
    "Remote code execution in image parser"
)
print(f"Severity: {severity}, Confidence: {confidence:.3f}")

# Get all class scores
severity, confidence, all_scores = inference.predict_severity(
    "Buffer overflow vulnerability", 
    return_all_scores=True
)
print(f"All scores: {all_scores}")
```

## 🎯 Severity Classes

The model predicts four severity levels:

| Severity | Description | Examples |
|----------|-------------|----------|
| **CRITICAL** | Remote code execution, privilege escalation | Buffer overflows, RCE vulnerabilities |
| **IMPORTANT** | Significant security impact | XSS, SQL injection, authentication bypass |
| **MODERATE** | Medium security risk | Information disclosure, DoS |
| **LOW** | Minor security issues | Configuration issues, version disclosure |

## 📊 Model Performance

- **Model**: Fine-tuned SecBERT on RedHat VeX dataset
- **Dataset Size**: 49k+ vulnerability records
- **Expected Accuracy**: >90%
- **Classes**: 4 severity levels (critical, important, moderate, low)

## 🔧 Installation & Requirements

```bash
# Required packages
pip install torch transformers numpy

# Optional for enhanced features
pip install matplotlib seaborn  # for visualizations
```

## 🖥️ Device Support

- **CUDA GPU**: Fastest inference
- **Apple Silicon MPS**: Optimized for M1/M2/M3 Macs
- **CPU**: Universal compatibility

The script automatically detects and uses the best available device.

## 📁 Files Description

| File | Purpose |
|------|---------|
| `secbert_inference.py` | Main inference script with CLI and API |
| `test_descriptions.txt` | Sample vulnerability descriptions for testing |
| `README.md` | This documentation |

## 🧪 Testing Your Setup

```bash

# Test with sample descriptions
python secbert_inference.py --file test_descriptions.txt

# Interactive testing
python secbert_inference.py
```

## 🔍 Command Line Arguments

```bash
python secbert_inference.py [OPTIONS]

Options:
  -t, --text TEXT     Single vulnerability description to classify
  -f, --file FILE     File with descriptions (one per line)
  -j, --json         Output in JSON format
  -m, --model MODEL  Custom HuggingFace model name
  -h, --help         Show help message

Examples:
  python secbert_inference.py
  python secbert_inference.py --text "XSS vulnerability"
  python secbert_inference.py --file vuln_list.txt --json
```

## 🚨 Error Handling

The inference script includes robust error handling for:

- **Network Issues**: Automatic retry for model download
- **Device Compatibility**: Falls back from MPS→CPU if needed
- **Input Validation**: Handles empty or malformed inputs
- **Model Loading**: Clear error messages for troubleshooting

## 🔄 Integration Examples

### Web API Integration
```python
from flask import Flask, request, jsonify
from secbert_inference import SecBERTInference

app = Flask(__name__)
inference = SecBERTInference()
inference.load_model()

@app.route('/predict', methods=['POST'])
def predict():
    description = request.json['description']
    severity, confidence = inference.predict_severity(description)
    return jsonify({
        'severity': severity,
        'confidence': confidence
    })
```

### Batch Processing Script
```python
import pandas as pd
from secbert_inference import SecBERTInference

# Load vulnerability data
df = pd.read_csv('vulnerabilities.csv')

# Initialize inference
inference = SecBERTInference()
inference.load_model()

# Add predictions
results = []
for desc in df['description']:
    severity, confidence = inference.predict_severity(desc)
    results.append({'severity': severity, 'confidence': confidence})

# Save results
df_results = pd.DataFrame(results)
df_combined = pd.concat([df, df_results], axis=1)
df_combined.to_csv('vulnerabilities_with_predictions.csv', index=False)
```

## 📈 Performance Tips

1. **Batch Processing**: Use `predict_batch()` for multiple descriptions
2. **GPU Acceleration**: Use CUDA/MPS for faster inference on large datasets
3. **Text Length**: Optimal performance with descriptions under 512 tokens
4. **Model Caching**: Model loads once and can predict many times efficiently

## 🤝 Contributing

To improve the inference capabilities:

1. Test with your own vulnerability descriptions
2. Report any accuracy issues or edge cases
3. Suggest additional output formats or integrations
4. Share performance benchmarks on different hardware

---

**Model Source**: [huzaifas-sidhpurwala/secbert-redhat-data](https://huggingface.co/huzaifas-sidhpurwala/secbert-redhat-data)  
**Dataset**: [RedHat Security VeX](https://huggingface.co/datasets/huzaifas-sidhpurwala/RedHat-security-VeX)
