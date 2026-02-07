from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
import matplotlib.pyplot as plt
import mathcoreml.utils.csvstore as csvstore
from mathcoreml.Models.template_models.modelImplementationTemplate import Models 
from mathcoreml.utils.cleandata import cleandata
from mathcoreml.utils.modelevalutaor import RegressionEvaluator
class linear_regression(Models):
    def __init__(self,object:csvstore):
        self.model = LinearRegression()
        self._df = object.frame
        self.is_trained = False
        self.is_preaper = False
    def preaper_data(self,x_column,y_column):
        """
        Prepare data for model training by separating features and target.
        
        Args:
            x_column: Column name to use as target variable (X)
            y_column: Column name to use as target variable (Y)
        
        Returns:
            None (prepares self.X and self.Y)
        """
        if self.is_preaper:
            print("Your Data is preapered")
            return 
        q1_x = self._df[x_column].quantile(0.25)
        q3_x = self._df[x_column].quantile(0.75)
        q1_y = self._df[y_column].quantile(0.25)
        q3_y = self._df[y_column].quantile(0.75)

        self.__clean_data = cleandata(q1_x, q3_x, q1_y, q3_y)
        print(self.__clean_data.x_lower_bound,self.__clean_data.x_upper_bound,self.__clean_data.y_lower_bound,self.__clean_data.y_upper_bound)
        mask_x = self._df[x_column].apply(lambda x: not self.__clean_data.x_is_outlier(x))
        mask_y = self._df[y_column].apply(lambda y: not self.__clean_data.y_is_outlier(y))
        self._df = self._df[mask_x & mask_y].copy()
        self._df = self._df.dropna(subset=[x_column, y_column])
        self._df = self._df.reset_index(drop=True)
        self.X = self._df[[x_column]]
        self.Y = self._df[y_column]
        self.is_preaper = True
    def train_model(self):
        """
        Train the  Lineary regression model using prepared data.
        
        Args:
            None
        
        Returns:
            float: Model accuracy score on test set
        """
        if self.is_trained:
            print("Your Model is trained")
            return
        X_train,X_test,Y_train,Y_test = train_test_split(self.X,self.Y,test_size=0.2,train_size=0.8)
        self.model.fit(X_train,Y_train)
        self.is_trained = True
        self.y_hat = self.model.predict(self.X)
        return self.model.score(X_test,Y_test)
    def predict_single(self, index):
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
        return self.model.predict(self.X.iloc[[index]])[0]
    def plot_regression(self):
        plt.figure()
        plt.scatter(self.X,self.Y,color='black')
        self._slope = self.model.coef_[0]
        self.intercept = self.model.intercept_
        y_line = self._slope * self.X + self.intercept
        plt.plot(self.X,y_line,color='red',label="Regression line")
        plt.xlabel(self.X.columns[0])
        plt.ylabel(self.Y.name)
        plt.title(f"{self.X.columns[0]} and {self.Y.name} regressoin graph")
        plt.tight_layout()
        plt.show()
    def compare_dataset(self):
        evaluator = RegressionEvaluator(self.X,self.y_hat)
        evaluator.get_all_metrics()
        evaluator.print_metrics()
        ess ,rss,tss = evaluator.calculate_sum_of_squares()
        print(f"ESS:{ess}")
        print(f"RSS:{rss}")
        print(f"TSS:{tss}")

        