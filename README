# MathCoreML

Lightweight utilities and example models for small-scale machine learning experiments.

## Overview

**MathCoreML** is a compact Python library designed for learning, teaching, and rapid prototyping of small machine learning workflows.  
The project emphasizes **clarity, simplicity, and minimal dependencies**, making it suitable for educational use and lightweight experiments.

It provides:

- Simple CSV data exploration utilities
- Basic data cleaning helpers
- Minimal reference implementations of classic ML models

---

## Features

### 📊 Data Utilities

- **CSVStore**
    - Easy CSV loading
    - Basic statistics (min, max, mean, counts)
    - Quick summaries and simple visualizations
- **CleanData**
    - IQR-based outlier detection
    - Lightweight data-cleaning helpers

### 🤖 Models

- **Linear Regression**
- **Logistic Regression**

These implementations are intentionally minimal and readable, aimed at education and experimentation rather than production-scale systems.

---

## Installation

Install from PyPI (once published):

```bash
pip install mathcoreml
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

Example usage with CSVStore (provide your own CSV file):

```python
from MathCoreML.utils.CSVStore import CSVStore
from MathCoreML.utils.CleanData import CleanData

store = CSVStore('/absolute/path/to/your_dataset.csv')
print('Rows:', len(store))
print('Max Age:', store.max_of('Age'))

store.quick_summary()
```

## Package Structure

MathCoreML/
├─ src/
│ └─ MathCoreML/
│ ├─ utils/
│ │ ├─ CSVStore.py # CSV loading, stats, plots
│ │ ├─ CleanData.py # IQR-based outlier detection
│ │ └─ ModelEvaluation.py
│ └─ Models/
│ ├─ linearRegression.py
│ └─ logisticRegression.py

## Notes

### Designed for clarity and learning, not as a replacement for full ML frameworks.

#### Public API is limited to modules inside MathCoreML.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
