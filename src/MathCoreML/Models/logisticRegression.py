from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mathcoreml.utils.csvstore as csvstore
from mathcoreml.Models.template_models.modelImplementationTemplate import  Models
class logistic_regression(Models):
    def __init__(self,dataset_object:csvstore):
        """
        Initialize logistic regression model with a dataset.
        
        Args:
            dataset_object: csvstore object containing the dataset
        
        Returns:
            None
        """
        self.model = LogisticRegression(max_iter=1000)
        self._df = dataset_object.frame
        self.is_trained = False
        self.is_preaper = False
    def preaper_data(self,x_column):
        """
        Prepare data for model training by separating features and target.
        
        Args:
            x_column: Column name to use as target variable (Y)
        
        Returns:
            None (prepares self.X and self.Y)
        """
        if self.is_preaper:
            print("Your  Data is preapered")
            return 
        self.X = self._df.drop(x_column,axis=1)
        self.Y = self._df[x_column]
        self.is_preaper = True
    def train_model(self):
        """
        Train the logistic regression model using prepared data.
        
        Args:
            None
        
        Returns:
            float: Model accuracy score on test set
        """
        if self.is_trained:
            print("Your Model is tarined")
            return
        X_train , X_test , Y_train  , Y_test  = train_test_split(self.X,self.Y,test_size=0.2,train_size=0.7)
        self.model.fit(X_train,Y_train)
        self.is_trained = True
        return self.model.score(X_test,Y_test)
    def predict_single(self,index:int):
        """
        Predict the probability for a single sample at given index.
        
        Args:
            index: Row index in the dataset (0-based)
        
        Returns:
            float: Predicted probability of the positive class (0-1)
        """
        if index < 0  or index >  len(self._df):
            print("Invalid Index")
            return 0
        row_df = self.X.iloc[[index]]
        return self.model.predict_proba(row_df)[0][1]    
    def plot_my_model_sigmoid(self):
        """
        Visualize the logistic regression sigmoid curve with predictions.
        
        Args:
            None
        
        Returns:
            None (displays sigmoid curve plot)
        """
        z_values = self.model.decision_function(self.X)
        probs = self.model.predict_proba(self.X)[:, 1]
    
        z_range = np.linspace(z_values.min() - 1, z_values.max() + 1, 100)
        sigmoid_curve = 1 / (1 + np.exp(-z_range))
        
        plt.figure(figsize=(10, 6))
        plt.plot(z_range, sigmoid_curve, color='blue', alpha=0.5, label='Sigmoid Curve')
        
        plt.scatter(z_values, probs, c=probs, cmap='RdYlGn_r', edgecolors='black', s=40)
        
        plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.axvline(0, color='black', alpha=0.3)
        
        plt.title('My Logistic Regression Sigmoid')
        plt.xlabel('z')
        plt.ylabel('Probabilty')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()