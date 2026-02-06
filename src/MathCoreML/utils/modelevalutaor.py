"""Model evaluation metrics for regression and classification models."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
import seaborn as sns

class RegressionEvaluator:
    """Evaluate regression models with multiple metrics."""
    
    def __init__(self, y_true, y_pred):
        """
        Initialize RegressionEvaluator with actual and predicted values.
        
        Args:
            y_true: Actual target values (array-like)
            y_pred: Predicted target values (array-like)
        
        Returns:
            None
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = {}
    
    def calculate_mse(self):
        """ RSS (Residual Sum of Squares).
        Calculate Mean Squared Error.
        
        Args:
            None
        
        Returns:
            float: Mean Squared Error value
        """
        mse = mean_squared_error(self.y_true, self.y_pred)
        self.metrics['MSE'] = mse
        return mse
    
    def calculate_rmse(self):
        """
        Calculate Root Mean Squared Error.
        
        Args:
            None
        
        Returns:
            float: Root Mean Squared Error value
        """
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        self.metrics['RMSE'] = rmse
        return rmse
    
    def calculate_mae(self):
        """
        Calculate Mean Absolute Error.
        
        Args:
            None
        
        Returns:
            float: Mean Absolute Error value
        """
        mae = mean_absolute_error(self.y_true, self.y_pred)
        self.metrics['MAE'] = mae
        return mae
    
    def calculate_r2(self):
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            None
        
        Returns:
            float: R² score (0-1, higher is better)
        """
        r2 = r2_score(self.y_true, self.y_pred)
        self.metrics['R²'] = r2
        return r2
    
    def calculate_adjusted_r2(self, n_features):
        """
        Calculate Adjusted R-squared.
        
        Args:
            n_features: Number of features used in the model
        
        Returns:
            float: Adjusted R² score
        """
        n = len(self.y_true)
        r2 = r2_score(self.y_true, self.y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
        self.metrics['Adjusted R²'] = adj_r2
        return adj_r2
    
    def get_all_metrics(self, n_features=1):
        """
        Calculate and return all regression metrics.
        
        Args:
            n_features: Number of features (default: 1 for simple regression)
        
        Returns:
            dict: Dictionary containing all metrics
        """
        self.calculate_mse()
        self.calculate_rmse()
        self.calculate_mae()
        self.calculate_r2()
        self.calculate_adjusted_r2(n_features)
        return self.metrics
    
    def print_metrics(self, n_features=1):
        """
        Print all metrics in a formatted way.
        
        Args:
            n_features: Number of features (default: 1)
        
        Returns:
            None (prints to console)
        """
        metrics = self.get_all_metrics(n_features)
        print("REGRESSION MODEL EVALUATION METRICS")
        for metric_name, value in metrics.items():
            print(f"{metric_name:20s}: {value:.6f}")
    
    def plot_residuals(self, title="Residual Plot"):
        """
        Plot residuals to visualize model fit.
        
        Args:
            title: Title for the plot
        
        Returns:
            None (displays plot)
        """
        residuals = self.y_true - self.y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_pred, residuals, alpha=0.6, edgecolors='k')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    def calculate_sum_of_squares(self):
        y_mean = np.mean(self.y_true)
        ess = np.sum((self.y_pred - y_mean)**2)
        rss = np.sum((self.y_true-y_mean)**2)
        tss = np.sum((self.y_true - y_mean)**2)
        return ess ,rss,tss
class ClassificationEvaluator:
    """Evaluate classification models with multiple metrics."""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize ClassificationEvaluator with actual and predicted values.
        
        Args:
            y_true: Actual target values (array-like)
            y_pred: Predicted class labels (array-like)
            y_pred_proba: Predicted probabilities for ROC curve (optional)
        
        Returns:
            None
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.metrics = {}
    
    def calculate_accuracy(self):
        """
        Calculate Accuracy score.
        
        Args:
            None
        
        Returns:
            float: Accuracy value (0-1)
        """
        accuracy = accuracy_score(self.y_true, self.y_pred)
        self.metrics['Accuracy'] = accuracy
        return accuracy
    
    def calculate_precision(self):
        """
        Calculate Precision score.
        
        Args:
            None
        
        Returns:
            float: Precision value (0-1)
        """
        precision = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        self.metrics['Precision'] = precision
        return precision
    
    def calculate_recall(self):
        """
        Calculate Recall score.
        
        Args:
            None
        
        Returns:
            float: Recall value (0-1)
        """
        recall = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        self.metrics['Recall'] = recall
        return recall
    
    def calculate_f1(self):
        """
        Calculate F1-score.
        
        Args:
            None
        
        Returns:
            float: F1-score value (0-1)
        """
        f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        self.metrics['F1-Score'] = f1
        return f1
    
    def calculate_roc_auc(self):
        """
        Calculate ROC-AUC score (for binary classification).
        
        Args:
            None
        
        Returns:
            float: ROC-AUC score (0-1)
        """
        if self.y_pred_proba is None:
            return None
        try:
            roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
            self.metrics['ROC-AUC'] = roc_auc
            return roc_auc
        except:
            return None
    
    def get_confusion_matrix(self):
        """
        Get the confusion matrix.
        
        Args:
            None
        
        Returns:
            np.ndarray: Confusion matrix
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        return cm
    
    def get_all_metrics(self):
        """
        Calculate and return all classification metrics.
        
        Args:
            None
        
        Returns:
            dict: Dictionary containing all metrics
        """
        self.calculate_accuracy()
        self.calculate_precision()
        self.calculate_recall()
        self.calculate_f1()
        self.calculate_roc_auc()
        return self.metrics
    
    def print_metrics(self):
        """
        Print all metrics in a formatted way.
        
        Args:
            None
        
        Returns:
            None (prints to console)
        """
        metrics = self.get_all_metrics()
        print("\n" + "="*50)
        print("CLASSIFICATION MODEL EVALUATION METRICS")
        print("="*50)
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"{metric_name:20s}: {value:.6f}")
        print("="*50)
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(self.y_true, self.y_pred, zero_division=0))
        print("="*50 + "\n")
    
    def plot_confusion_matrix(self, title="Confusion Matrix"):
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            title: Title for the plot
        
        Returns:
            None (displays plot)
        """
        cm = self.get_confusion_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
    
    def plot_roc_curve(self, title="ROC Curve"):
        """
        Plot ROC curve (for binary classification).
        
        Args:
            title: Title for the plot
        
        Returns:
            None (displays plot)
        """
        if self.y_pred_proba is None:
            print("Predicted probabilities not provided")
            return
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


class ModelComparison:
    """Compare multiple models."""
    
    def __init__(self):
        """
        Initialize ModelComparison.
        
        Args:
            None
        
        Returns:
            None
        """
        self.models = {}
    
    def add_regression_model(self, name, y_true, y_pred, n_features=1):
        """
        Add a regression model for comparison.
        
        Args:
            name: Model name (str)
            y_true: Actual values
            y_pred: Predicted values
            n_features: Number of features
        
        Returns:
            None
        """
        evaluator = RegressionEvaluator(y_true, y_pred)
        metrics = evaluator.get_all_metrics(n_features)
        self.models[name] = {'type': 'regression', 'metrics': metrics}
    
    def add_classification_model(self, name, y_true, y_pred, y_pred_proba=None):
        """
        Add a classification model for comparison.
        
        Args:
            name: Model name (str)
            y_true: Actual values
            y_pred: Predicted values
            y_pred_proba: Predicted probabilities (optional)
        
        Returns:
            None
        """
        evaluator = ClassificationEvaluator(y_true, y_pred, y_pred_proba)
        metrics = evaluator.get_all_metrics()
        self.models[name] = {'type': 'classification', 'metrics': metrics}
    
    def compare_models(self, metric_name):
        """
        Compare models on a specific metric.
        
        Args:
            metric_name: Name of the metric to compare
        
        Returns:
            dict: Dictionary with model names and metric values
        """
        comparison = {}
        for model_name, model_data in self.models.items():
            if metric_name in model_data['metrics']:
                comparison[model_name] = model_data['metrics'][metric_name]
        return comparison
    
    def plot_comparison(self, metric_name, title=None):
        """
        Plot model comparison for a specific metric.
        
        Args:
            metric_name: Name of the metric to compare
            title: Custom title (optional)
        
        Returns:
            None (displays plot)
        """
        comparison = self.compare_models(metric_name)
        if not comparison:
            print(f"No data for metric: {metric_name}")
            return
        
        plt.figure(figsize=(10, 6))
        models = list(comparison.keys())
        values = list(comparison.values())
        
        plt.bar(models, values, color='steelblue', edgecolor='black')
        plt.ylabel(metric_name)
        plt.title(title or f"Model Comparison: {metric_name}")
        plt.ylim([0, 1.0] if metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'R²'] else None)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
    
    def summary(self):
        """
        Print summary of all models and their metrics.
        
        Args:
            None
        
        Returns:
            None (prints to console)
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        for model_name, model_data in self.models.items():
            print(f"\n{model_name} ({model_data['type'].upper()}):")
            for metric_name, value in model_data['metrics'].items():
                if value is not None:
                    print(f"  {metric_name:20s}: {value:.6f}")
        print("\n" + "="*70 + "\n")
