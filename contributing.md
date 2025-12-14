# Contributing to Material Classification ANN

Thank you for your interest in contributing to this project! 🎉

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Commit Messages](#commit-messages)
7. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

### Our Standards:
- ✅ Be welcoming and inclusive
- ✅ Respect differing viewpoints
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what's best for the community
- ❌ No harassment or discriminatory language
- ❌ No personal attacks or trolling

---

## How Can I Contribute?

### 🐛 Reporting Bugs

If you find a bug, please create an issue with:
- Clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Python version and OS
- Relevant code snippets or error messages

**Template:**
```markdown
**Bug Description:**
A clear description of the bug.

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. See error

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Environment:**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.8.10]
- Package Versions: [paste requirements.txt]

**Screenshots/Logs:**
If applicable, add screenshots or error logs.
```

### 💡 Suggesting Enhancements

Feature requests are welcome! Please include:
- Clear description of the enhancement
- Use cases and benefits
- Possible implementation approach
- Any relevant examples or references

### 📝 Improving Documentation

Documentation improvements are always appreciated:
- Fix typos or clarify confusing sections
- Add examples or tutorials
- Improve API documentation
- Translate documentation

### 🔧 Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Write/update tests**
5. **Update documentation**
6. **Submit a pull request**

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ann-material-classification.git
cd ann-material-classification
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### 4. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## Coding Standards

### Python Style Guide

Follow **PEP 8** guidelines:

```python
# Good: Clear variable names, proper spacing
def calculate_accuracy(predictions, labels):
    """Calculate classification accuracy.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    labels : np.ndarray
        True labels
        
    Returns
    -------
    float
        Accuracy score between 0 and 1
    """
    correct = np.sum(predictions == labels)
    total = len(labels)
    return correct / total

# Bad: Unclear names, no docstring
def calc(p,l):
    return np.sum(p==l)/len(l)
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Tuple, Optional
import numpy as np

def preprocess_data(
    X: np.ndarray, 
    y: np.ndarray,
    test_size: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess and split data."""
    # Implementation
    return X_train, X_test, y_train, y_test
```

### Docstrings

Use NumPy-style docstrings:

```python
def train_model(X_train, y_train, hidden_layers=(10,), learning_rate=0.001):
    """Train ANN model on training data.
    
    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_features)
        Training feature matrix
    y_train : np.ndarray, shape (n_samples,)
        Training labels
    hidden_layers : tuple, default=(10,)
        Hidden layer configuration
    learning_rate : float, default=0.001
        Learning rate for optimizer
        
    Returns
    -------
    model : MLPClassifier
        Trained model
        
    Examples
    --------
    >>> model = train_model(X_train, y_train, hidden_layers=(20, 10))
    >>> accuracy = model.score(X_test, y_test)
    """
    pass
```

### Code Organization

```python
# 1. Standard library imports
import os
import sys
from typing import List, Dict

# 2. Third-party imports
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

# 3. Local imports
from .preprocessing import normalize_data
from .evaluation import calculate_metrics
```

---

## Testing Guidelines

### Writing Tests

Use **pytest** for testing:

```python
# tests/test_preprocessing.py
import pytest
import numpy as np
from src.preprocessing import normalize_data

def test_normalize_data_shape():
    """Test that normalization preserves shape."""
    X = np.random.rand(100, 2)
    X_norm = normalize_data(X)
    assert X_norm.shape == X.shape

def test_normalize_data_mean():
    """Test that normalized data has zero mean."""
    X = np.random.rand(100, 2)
    X_norm = normalize_data(X)
    np.testing.assert_almost_equal(X_norm.mean(axis=0), 0, decimal=10)

def test_normalize_data_std():
    """Test that normalized data has unit std."""
    X = np.random.rand(100, 2)
    X_norm = normalize_data(X)
    np.testing.assert_almost_equal(X_norm.std(axis=0), 1, decimal=10)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run specific test
pytest tests/test_preprocessing.py::test_normalize_data_shape
```

### Test Coverage

Aim for **>80% code coverage**:

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html in browser
```

---

## Commit Messages

Follow **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples:

```bash
# Good commit messages
git commit -m "feat(model): add support for custom activation functions"
git commit -m "fix(preprocessing): handle missing values in dataset"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(evaluation): add tests for confusion matrix calculation"

# Bad commit messages
git commit -m "fixed bug"
git commit -m "changes"
git commit -m "update"
```

### Detailed Example:

```
feat(visualization): add interactive decision boundary plot

- Implement interactive plotly visualization
- Add hover tooltips showing material names
- Include zoom and pan functionality
- Update documentation with usage examples

Closes #42
```

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clean, documented code
- Add/update tests
- Update documentation
- Ensure all tests pass

### 3. Commit Changes

```bash
git add .
git commit -m "feat(scope): description"
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request

**PR Template:**

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Testing
- [ ] All existing tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated

## Related Issues
Fixes #(issue number)

## Screenshots (if applicable)
Add screenshots here.
```

### 6. Code Review

- Respond to reviewer comments
- Make requested changes
- Push updates to the same branch

### 7. Merge

Once approved:
- Maintainer will merge your PR
- Delete your feature branch

---

## Development Workflow

### Branching Strategy

```
main
  ├── develop
  │   ├── feature/new-feature
  │   ├── fix/bug-fix
  │   └── docs/update-readme
  └── release/v1.0.0
```

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates
- `release/*`: Release preparation

### Release Process

1. Create release branch from `develop`
2. Update version numbers
3. Update CHANGELOG.md
4. Final testing
5. Merge to `main` and tag
6. Merge back to `develop`

---

## Additional Resources

### Learning Resources
- [PEP 8 Style Guide](https://pep8.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

### Project Documentation
- [API Reference](docs/api_reference.md)
- [Methodology](docs/methodology.md)
- [Results](docs/results.md)

### Communication
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: [Contact information]

---

## Questions?

If you have questions:
1. Check existing issues and discussions
2. Review documentation
3. Create a new issue with the `question` label

---

Thank you for contributing! 🙏

**Happy Coding!** 🚀