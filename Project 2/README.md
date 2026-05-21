# Image Classification Project
## Comparative Study: Naive Bayes vs Decision Tree vs Neural Network

This project implements and compares three different machine learning models for image classification:
1. **Naive Bayes Classifier** - Fast and effective for high-dimensional data
2. **Decision Tree Classifier** - Interpretable and suitable for rule-based classification
3. **Feedforward Neural Network (MLPClassifier)** - Deep learning approach for complex patterns

### Project Requirements
- Python 3.7+
- At least 500 labeled images organized into 3+ categories
- Images will be resized to 32×32 pixels and flattened into 1D feature vectors

### Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Dataset:**
   - Create a folder named `dataset` in the project directory
   - Inside `dataset`, create subfolders for each class (e.g., `cat`, `dog`, `bird`)
   - Place images of each class in their respective folders
   
   Example structure:
   ```
   dataset/
   ├── cat/
   │   ├── cat1.jpg
   │   ├── cat2.jpg
   │   └── ...
   ├── dog/
   │   ├── dog1.jpg
   │   ├── dog2.jpg
   │   └── ...
   └── bird/
       ├── bird1.jpg
       ├── bird2.jpg
       └── ...
   ```

3. **Run the Analysis:**
   ```bash
   python main.py
   ```

### What the Program Does

1. **Data Preprocessing:**
   - Loads images from the dataset folder
   - Resizes all images to 32×32 pixels
   - Converts to RGB format and flattens into 1D vectors
   - Splits data into training (80%) and testing (20%) sets

2. **Model Training:**
   - **Naive Bayes:** Uses GaussianNB for pixel-level classification
   - **Decision Tree:** Builds interpretable decision rules (max depth=10)
   - **Neural Network:** MLPClassifier with 2 hidden layers (100, 50 neurons)

3. **Evaluation:**
   - Calculates accuracy, precision, recall, and F1-score
   - Generates confusion matrices
   - Performs 5-fold cross-validation
   - Creates visualizations and comparison tables

### Output Files

The program generates several output files:
- `confusion_matrices.png` - Confusion matrices for all models
- `decision_tree.png` - Visualization of decision tree structure
- `cross_validation_results.png` - Cross-validation performance comparison
- `model_comparison.csv` - Detailed metrics comparison table

### Customization Options

You can modify the following parameters in `main.py`:

```python
# Image size (default: 32x32)
classifier = ImageClassifier(image_size=(64, 64))

# Test split ratio (default: 0.2)
classifier.prepare_data(data_folder, test_size=0.3)

# Decision tree depth (default: 10)
classifier.train_decision_tree(max_depth=15)

# Neural network architecture (default: (100, 50))
classifier.train_neural_network(hidden_layer_sizes=(200, 100, 50))
```

### Model Characteristics

**Naive Bayes:**
- ✅ Fast training and prediction
- ✅ Works well with high-dimensional data
- ✅ No hyperparameter tuning needed
- ❌ Assumes feature independence
- ❌ May not capture complex patterns

**Decision Tree:**
- ✅ Highly interpretable
- ✅ Can handle non-linear relationships
- ✅ No feature scaling required
- ❌ Can overfit easily
- ❌ Sensitive to small data changes

**Neural Network:**
- ✅ Can learn complex patterns
- ✅ Good generalization with proper regularization
- ✅ Handles non-linear relationships well
- ❌ Requires more data and training time
- ❌ Less interpretable
- ❌ Needs feature scaling

### Troubleshooting

**Common Issues:**
1. **"Data folder not found"** - Make sure your dataset folder is named `dataset` and is in the same directory as `main.py`
2. **Memory errors** - Reduce image size or use fewer images
3. **Slow training** - Reduce neural network complexity or use fewer iterations

**Performance Tips:**
- Use SSD storage for faster image loading
- Ensure balanced class distribution
- Consider using grayscale images for faster processing 