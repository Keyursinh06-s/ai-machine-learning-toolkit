# Contributing to AI Machine Learning Toolkit

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in Issues
- Use the bug report template
- Include detailed steps to reproduce
- Provide system information and error messages

### Suggesting Features

- Check if the feature has been suggested
- Use the feature request template
- Explain the use case and benefits
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new features
   - Update documentation

4. **Commit your changes**
   ```bash
   git commit -m "Add: Brief description of changes"
   ```
   
   Commit message format:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Docs:` for documentation changes
   - `Test:` for test additions/changes

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Use the PR template
   - Link related issues
   - Provide clear description of changes

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Keyursinh06-s/ai-machine-learning-toolkit.git
cd ai-machine-learning-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## Code Style

### Python

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all functions and classes

Example:
```python
def process_data(data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process input data with optional normalization.
    
    Args:
        data: Input data array
        normalize: Whether to normalize the data
        
    Returns:
        Processed data array
    """
    # Implementation
    pass
```

### Documentation

- Update README.md for significant changes
- Add docstrings to all public functions
- Include usage examples
- Update API documentation

### Testing

- Write unit tests for new features
- Maintain test coverage above 80%
- Use pytest for testing
- Mock external dependencies

Example test:
```python
def test_data_preprocessing():
    """Test data preprocessing functionality"""
    preprocessor = DataPreprocessor()
    data = np.random.rand(100, 10)
    result = preprocessor.process(data)
    assert result.shape == data.shape
```

## Project Structure

```
src/
├── neural_networks/    # Neural network implementations
├── data_processing/    # Data preprocessing utilities
├── deployment/         # Model deployment tools
├── visualization/      # Visualization utilities
└── utils/             # Helper functions

tests/                 # Test files
docs/                  # Documentation
examples/              # Usage examples
```

## Review Process

1. Automated checks must pass (linting, tests)
2. Code review by maintainers
3. Address feedback and update PR
4. Approval and merge

## Questions?

- Open an issue for questions
- Join our community discussions
- Check existing documentation

Thank you for contributing!