import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImageClassifier:
    def __init__(self, image_size=(32, 32)):
        """
        Initialize the Image Classifier with specified image size.
        
        Args:
            image_size (tuple): Target size for images (width, height)
        """
        self.image_size = image_size
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = None
        
    def extract_statistical_features(self, image_array):
        """
        Extract statistical summaries (mean, variance) from image array.
        This provides an alternative feature representation for Naive Bayes.
        
        Args:
            image_array (array): 3D image array (height, width, channels)
            
        Returns:
            array: Statistical features (mean and variance for each channel)
        """
        # Calculate mean and variance for each RGB channel
        means = np.mean(image_array, axis=(0, 1))  # Mean for each channel
        variances = np.var(image_array, axis=(0, 1))  # Variance for each channel
        
        # Combine means and variances
        statistical_features = np.concatenate([means, variances])
        
        return statistical_features
    
    def load_and_preprocess_images(self, data_folder, max_images_per_class=None, use_statistical_features=False):
        """
        Load images from folder structure and preprocess them.
        
        Args:
            data_folder (str): Path to the folder containing class subfolders
            max_images_per_class (int): Maximum number of images to load per class (for balancing)
            use_statistical_features (bool): Whether to use statistical summaries instead of raw pixels
            
        Returns:
            tuple: (features, labels, class_names)
        """
        print("Loading and preprocessing images...")
        
        features = []
        labels = []
        class_names = []
        class_counts = {}
        
        # Define the classes we want to use
        target_classes = ['Big Truck', 'Van', 'City Car']
        
        # Walk through the data folder
        for class_name in os.listdir(data_folder):
            class_path = os.path.join(data_folder, class_name)
            
            # Skip if not a directory or not in our target classes
            if not os.path.isdir(class_path) or class_name not in target_classes:
                continue
                
            class_names.append(class_name)
            class_counts[class_name] = 0
            print(f"Processing class: {class_name}")
            
            # Get all image files in the class folder
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Limit images per class if specified
            if max_images_per_class and len(image_files) > max_images_per_class:
                import random
                random.seed(42)  # For reproducibility
                image_files = random.sample(image_files, max_images_per_class)
                print(f"  Limited to {max_images_per_class} images (randomly sampled)")
            
            # Process each image in the class folder
            for image_name in image_files:
                image_path = os.path.join(class_path, image_name)
                
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize(self.image_size)
                    
                    # Convert to numpy array
                    image_array = np.array(image)
                    
                    if use_statistical_features:
                        # Extract statistical summaries (mean, variance for each channel)
                        feature_vector = self.extract_statistical_features(image_array)
                        print(f"  Using statistical features: {len(feature_vector)} features (mean+var per channel)")
                    else:
                        # Flatten raw pixel values
                        feature_vector = image_array.flatten()
                    
                    features.append(feature_vector)
                    labels.append(class_name)
                    class_counts[class_name] += 1
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        print(f"\nFinal dataset summary:")
        for class_name in class_names:
            print(f"  {class_name}: {class_counts[class_name]} images")
        
        total_images = sum(class_counts.values())
        print(f"Total: {total_images} images from {len(class_names)} classes")
        
        if use_statistical_features:
            print(f"Feature vector size: {len(features[0])} (statistical summaries)")
        else:
            print(f"Feature vector size: {len(features[0])} (flattened {self.image_size[0]}x{self.image_size[1]}x3)")
        
        return np.array(features), np.array(labels), class_names
    
    def prepare_data(self, data_folder, test_size=0.2, random_state=42, max_images_per_class=None, use_statistical_features=False):
        """
        Prepare the dataset by loading, preprocessing, and splitting into train/test sets.
        
        Args:
            data_folder (str): Path to the data folder
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            max_images_per_class (int): Maximum number of images per class (for balancing)
            use_statistical_features (bool): Whether to use statistical summaries instead of raw pixels
        """
        # Load and preprocess images
        X, y, self.class_names = self.load_and_preprocess_images(data_folder, max_images_per_class, use_statistical_features)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Classes: {self.class_names}")
        
        # Scale features for neural network
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def train_naive_bayes(self):
        """Train Naive Bayes classifier."""
        print("\nTraining Naive Bayes Classifier...")
        nb_model = GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        self.models['naive_bayes'] = nb_model
        print("Naive Bayes training completed!")
    
    def train_decision_tree(self, max_depth=None, random_state=42):
        """
        Train Decision Tree classifier.
        
        Args:
            max_depth (int): Maximum depth of the tree
            random_state (int): Random seed for reproducibility
        """
        print("\nTraining Decision Tree Classifier...")
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        dt_model.fit(self.X_train, self.y_train)
        self.models['decision_tree'] = dt_model
        print("Decision Tree training completed!")
    
    def train_neural_network(self, hidden_layer_sizes=(100, 50), max_iter=500, random_state=42):
        """
        Train Feedforward Neural Network (MLPClassifier).
        
        Args:
            hidden_layer_sizes (tuple): Number of neurons in each hidden layer
            max_iter (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility
        """
        print("\nTraining Feedforward Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            verbose=True
        )
        nn_model.fit(self.X_train_scaled, self.y_train)
        self.models['neural_network'] = nn_model
        print("Neural Network training completed!")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a specific model and return metrics.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (array): Test features
            y_test (array): Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        model = self.models[model_name]
        
        # Make predictions
        if model_name == 'neural_network':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models and return results."""
        results = {}
        
        for model_name in self.models.keys():
            print(f"\nEvaluating {model_name.replace('_', ' ').title()}...")
            
            if model_name == 'neural_network':
                results[model_name] = self.evaluate_model(
                    model_name, self.X_test_scaled, self.y_test
                )
            else:
                results[model_name] = self.evaluate_model(
                    model_name, self.X_test, self.y_test
                )
            
            # Print results
            metrics = results[model_name]
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
        
        if len(results) == 1:
            axes = [axes]
        
        for i, (model_name, metrics) in enumerate(results.items()):
            conf_matrix = metrics['confusion_matrix']
            
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[i]
            )
            axes[i].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_decision_tree(self, max_depth=3):
        """Plot a simplified decision tree (limited depth for visualization)."""
        if 'decision_tree' not in self.models:
            print("Decision tree not trained yet!")
            return
        
        # Create a simplified tree for visualization
        simple_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        simple_tree.fit(self.X_train, self.y_train)
        
        plt.figure(figsize=(20, 10))
        plot_tree(
            simple_tree,
            feature_names=[f'pixel_{i}' for i in range(self.X_train.shape[1])],
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=8
        )
        plt.title(f'Decision Tree Visualization (Max Depth: {max_depth})')
        plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results):
        """Create a comparison table of all models."""
        comparison_data = []
        
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        # Save comparison to CSV
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("\nComparison saved to 'model_comparison.csv'")
        
        return comparison_df
    
    def cross_validation_analysis(self, cv=5):
        """Perform cross-validation analysis for all models."""
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                X_data = self.X_train_scaled
            else:
                X_data = self.X_train
            
            cv_scores = cross_val_score(model, X_data, self.y_train, cv=cv, scoring='accuracy')
            cv_results[model_name] = cv_scores
            
            print(f"{model_name.replace('_', ' ').title()}:")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Plot CV results
        plt.figure(figsize=(10, 6))
        model_names = [name.replace('_', ' ').title() for name in cv_results.keys()]
        cv_means = [scores.mean() for scores in cv_results.values()]
        cv_stds = [scores.std() for scores in cv_results.values()]
        
        plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        plt.title(f'{cv}-Fold Cross-Validation Results')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            plt.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cv_results
    
    def compare_naive_bayes_features(self):
        """
        Compare Naive Bayes performance with pixel-level features vs statistical summaries.
        This demonstrates the baseline model evaluation as mentioned in project requirements.
        """
        print("\n" + "="*60)
        print("NAIVE BAYES FEATURE COMPARISON")
        print("="*60)
        
        # Test with pixel-level features (current approach)
        print("\n1. Testing with Pixel-Level Features:")
        classifier_pixel = ImageClassifier(self.image_size)
        classifier_pixel.prepare_data("Image", use_statistical_features=False)
        classifier_pixel.train_naive_bayes()
        results_pixel = classifier_pixel.evaluate_all_models()
        
        # Test with statistical summaries
        print("\n2. Testing with Statistical Summaries (Mean, Variance):")
        classifier_stats = ImageClassifier(self.image_size)
        classifier_stats.prepare_data("Image", use_statistical_features=True)
        classifier_stats.train_naive_bayes()
        results_stats = classifier_stats.evaluate_all_models()
        
        # Compare results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print("Feature Type          | Accuracy | Precision | Recall | F1-Score")
        print("-" * 60)
        
        pixel_metrics = results_pixel['naive_bayes']
        stats_metrics = results_stats['naive_bayes']
        
        print(f"Pixel-Level Features  | {pixel_metrics['accuracy']:.4f} | {pixel_metrics['precision']:.4f} | {pixel_metrics['recall']:.4f} | {pixel_metrics['f1_score']:.4f}")
        print(f"Statistical Summaries | {stats_metrics['accuracy']:.4f} | {stats_metrics['precision']:.4f} | {stats_metrics['recall']:.4f} | {stats_metrics['f1_score']:.4f}")
        
        # Determine which performs better
        if pixel_metrics['accuracy'] > stats_metrics['accuracy']:
            print(f"\nPixel-level features perform better by {pixel_metrics['accuracy'] - stats_metrics['accuracy']:.4f} accuracy")
        else:
            print(f"\nStatistical summaries perform better by {stats_metrics['accuracy'] - pixel_metrics['accuracy']:.4f} accuracy")
        
        return results_pixel, results_stats

def main():
    """Main function to run the complete image classification pipeline."""
    print("="*60)
    print("IMAGE CLASSIFICATION PROJECT")
    print("Comparative Study: Naive Bayes vs Decision Tree vs Neural Network")
    print("Dataset: Big Trucks vs Vans vs City Cars")
    print("="*60)
    
    # Initialize the classifier
    classifier = ImageClassifier(image_size=(32, 32))
    
    # Check if data folder exists
    data_folder = "Image"  # Updated to use your Image folder
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' not found!")
        print("Please make sure your image dataset folder is in the current directory.")
        return
    
    # Show dataset options
    print("\nDataset Options:")
    print("1. Use all available images (7,401+ images, may be slow)")
    print("2. Use balanced dataset (500 images per class = 1,500 total)")
    print("3. Use moderate balanced dataset (200 images per class = 600 total)")
    print("4. Quick test with small dataset (100 images per class = 300 total)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Set max_images_per_class based on choice
    if choice == '1':
        max_images_per_class = None  # Use all images
        print("\nUsing all available images (this may take a while)...")
    elif choice == '2':
        max_images_per_class = 500
        print("\nUsing balanced dataset with 500 images per class...")
    elif choice == '3':
        max_images_per_class = 200
        print("\nUsing moderate balanced dataset with 200 images per class...")
    elif choice == '4':
        max_images_per_class = 100
        print("\nUsing quick test dataset with 100 images per class...")
    
    # Prepare the data
    classifier.prepare_data(data_folder, test_size=0.2, max_images_per_class=max_images_per_class)
    
    # Train all models
    print("\n" + "="*40)
    print("TRAINING MODELS")
    print("="*40)
    
    classifier.train_naive_bayes()
    classifier.train_decision_tree(max_depth=10)
    classifier.train_neural_network(hidden_layer_sizes=(100, 50), max_iter=300)
    
    # Evaluate all models
    print("\n" + "="*40)
    print("EVALUATING MODELS")
    print("="*40)
    
    results = classifier.evaluate_all_models()
    
    # Generate visualizations and comparisons
    print("\n" + "="*40)
    print("GENERATING VISUALIZATIONS")
    print("="*40)
    
    classifier.plot_confusion_matrices(results)
    classifier.plot_decision_tree(max_depth=3)
    classifier.compare_models(results)
    classifier.cross_validation_analysis(cv=5)
    
    # Compare Naive Bayes with pixel-level features vs statistical summaries
    classifier.compare_naive_bayes_features()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("- confusion_matrices.png")
    print("- decision_tree.png")
    print("- cross_validation_results.png")
    print("- model_comparison.csv")

if __name__ == "__main__":
    main()
