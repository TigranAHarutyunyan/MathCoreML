"""Utilities for loading customer CSV data with pandas and numpy."""

from __future__ import annotations
from pathlib import Path
from typing import  Iterator
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from pandas.api.types import is_numeric_dtype
from mpl_toolkits.mplot3d import Axes3D
class csvstore:
    """Materializes a CSV file as a pandas DataFrame and numpy array."""
    def __init__(self, csv_path: str | Path):
        """
        Initialize csvstore with a CSV file path.
        
        Args:
            csv_path: Path to the CSV file (str or Path object)
        
        Returns:
            None
        
        Raises:
            FileNotFoundError: If the CSV file does not exist
        """
        self._csv_path = Path(csv_path)
        if not self._csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self._frame = self._read_frame()
    def _read_frame(self) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.
        
        Args:
            None
        
        Returns:
            pd.DataFrame: The CSV data loaded as a DataFrame
        """
        return pd.read_csv(self._csv_path)
    @property
    def csv_path(self) -> Path:
        """
        Get the path to the CSV file.
        
        Args:
            None
        
        Returns:
            Path: The Path object of the CSV file
        """
        return self._csv_path

    @property
    def frame(self) -> pd.DataFrame:
        """
        Get a copy of the DataFrame.
        
        Args:
            None
        
        Returns:
            pd.DataFrame: A copy of the internal DataFrame
        """
        return self._frame.copy()

    @property
    def as_numpy(self) -> np.ndarray:
        """
        Get the DataFrame as a numpy array.
        
        Args:
            None
        
        Returns:
            np.ndarray: A copy of the DataFrame as a numpy array
        """
        return self._frame.to_numpy(copy=True)
    def mean_of(self, column: str) -> float:
        """
        Calculate the mean of a column.
        
        Args:
            column: Name of the column to calculate mean for
        
        Returns:
            float: The mean value of the column
        """
        return self._frame[column].mean()
    def mode_of(self, column: str) -> float:
        """
        Calculate the mode of a column.
        
        Args:
            column: Name of the column to calculate mode for
        
        Returns:
            float: The mode value(s) of the column
        """
        return self._frame[column].mode()
    def head_of(self, quantity: int = 5) -> pd.DataFrame:
        """
        Get the first N rows of the DataFrame.
        
        Args:
            quantity: Number of rows to return (default: 5)
        
        Returns:
            pd.DataFrame: First N rows of the DataFrame
        """
        return self._frame.head(quantity)
    def std_of(self, column: str) -> float:
        """
        Calculate the standard deviation of a column.
        
        Args:
            column: Name of the column to calculate standard deviation for
        
        Returns:
            float: The standard deviation of the column
        """
        return self._frame[column].std()
    def min_of(self, column: str) -> float:
        """
        Get the minimum value of a column.
        
        Args:
            column: Name of the column
        
        Returns:
            float: The minimum value in the column
        """
        return self._frame[column].min()
    def max_of(self, column: str) -> float:
        """
        Get the maximum value of a column.
        
        Args:
            column: Name of the column
        
        Returns:
            float: The maximum value in the column
        """
        return self._frame[column].max()
    def count_where_equals(self, column: str, value) -> int:
        """
        Count rows where a column equals a specific value.
        
        Args:
            column: Name of the column to check
            value: The value to match
        
        Returns:
            int: Count of rows matching the condition
        """
        return (self._frame[column] == value).sum()
    def sturges(self) -> int:
        """
        Calculate Sturges' formula for optimal bin count.
        
        The Sturges' rule is a method for determining the number of bins in a histogram.
        Formula: k = ceil(log2(n) + 1) where n is the number of observations.
        
        Args:
            None
        
        Returns:
            int: The recommended number of bins using Sturges' formula
        """
        return int(1 + math.log2(len(self._frame)))
    def quick_summary(self) -> None:
        """
        Print a quick summary of the DataFrame.
        
        Displays basic information about the DataFrame including:
        - Column names, types, and non-null counts
        - Number of missing values per column
        - Descriptive statistics (count, mean, std, min, max, etc.)
        
        Args:
            None
        
        Returns:
            None (prints to console)
        """
        print("--- Basic Info ---")
        print(self._frame.info())
        print("\n--- Missing Values ---")
        print(self._frame.isna().sum())
        print("\n--- Descriptive Stats ---")
        print(self._frame.describe())
    def mean_of_group(self, x_column: str, y_column: str) -> pd.Series:
        """
        Calculate mean of y_column grouped by x_column.
        
        Groups the data by unique values in x_column and computes the mean
        of y_column for each group.
        
        Args:
            x_column: Column name to group by
            y_column: Column name to calculate mean for
        
        Returns:
            pd.Series: Mean values grouped by x_column
        """
        return self._frame.groupby(x_column, observed=True)[y_column].mean()
    def group_by(self, column: str) -> pd.Series:
        """
        Group a column into bins using Sturges' formula.
        
        Creates categorical bins for the specified column using Sturges' formula
        to determine the optimal number of bins. The binned data is stored as
        a new column in the DataFrame.
        
        Args:
            column: Column name to bin
        
        Returns:
            pd.Series: Binned categories for the column
        """
        age_bins = np.linspace(self.min_of(column), self.max_of(column), self.sturges())
        new_column: str = column + "Group"
        self._frame[new_column] = pd.cut(self._frame[column], age_bins)
        return self._frame[new_column]
    def histogram_of_one_column(self, x_column: str, title: str) -> None:
        """
        Create a histogram for a single column using matplotlib.
        
        Creates a histogram with automatic binning using Sturges' formula.
        
        Args:
            x_column: Column name to plot
            title: Title for the histogram
        
        Returns:
            None (displays plot)
        
        Raises:
            KeyError: If the specified column does not exist
        """
        if x_column not in self._frame.columns:
            raise KeyError(f"Column '{x_column}' not found")
        plt.figure()
        plt.hist(self._frame[x_column], bins='sturges', color="blue", edgecolor="black")
        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel("Frequency")
        plt.tight_layout()
    def histogram_of_one_column_sns(self, x_column: str, title: str) -> None:
        """
        Create a histogram for a single column using seaborn.
        
        Creates a histogram with KDE (Kernel Density Estimation) overlay
        for a smoother visualization of the distribution.
        
        Args:
            x_column: Column name to plot
            title: Title for the histogram
        
        Returns:
            None (displays plot with KDE)
        
        Raises:
            KeyError: If the specified column does not exist
        """
        if x_column not in self._frame.columns:
            raise KeyError(f"Column '{x_column}' not found")
        plt.figure()
        sns.histplot(self._frame[x_column], kde=True)
        plt.title(title)
    def scatter_of(self, x_column: str, y_column: str, title: str) -> None:
        """
        Create a scatter plot of two columns.
        
        Creates a scatter plot with the specified columns on x and y axes.
        Includes grid and axis labels.
        
        Args:
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            title: Title for the scatter plot
        
        Returns:
            None (displays plot)
        """
        plt.figure()
        plt.scatter(self._frame[x_column], self._frame[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title)
        plt.grid(True)
    def _prepare_numeric_pair(self, x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare two numeric columns for regression analysis.
        
        Args:
            x_column: First column name (must be numeric)
            y_column: Second column name (must be numeric)
        
        Returns:
            tuple: (x_values, y_values) as numpy arrays with NaN removed
        
        Raises:
            KeyError: If columns don't exist
            TypeError: If columns are not numeric
            ValueError: If insufficient valid data points
        """
        if x_column not in self._frame.columns or y_column not in self._frame.columns:
            missing = [col for col in (x_column, y_column) if col not in self._frame.columns]
            raise KeyError(f"Column(s) not found: {', '.join(missing)}")
        if not is_numeric_dtype(self._frame[x_column]) or not is_numeric_dtype(self._frame[y_column]):
            raise TypeError("Linear regression requires numeric columns")
        pair = self._frame[[x_column, y_column]].dropna()
        if len(pair) < 2:
            raise ValueError("Linear regression requires at least two valid rows")
        return pair[x_column].to_numpy(dtype=float), pair[y_column].to_numpy(dtype=float)
    def can_use_linear_regression(self, x_column: str, y_column: str, *,
                                  min_points: int = 3,
                                  ) -> bool:
        """
        Check if two columns are suitable for linear regression.
        
        Args:
            x_column: First column name
            y_column: Second column name
            min_points: Minimum data points required (default: 3)
        
        Returns:
            bool: True if R-squared >= 0.7, False otherwise
        """
        try:
            x_values, y_values = self._prepare_numeric_pair(x_column, y_column)
        except (KeyError, TypeError, ValueError):
            return False
        if len(x_values) < max(2, min_points):
            return False
        corr = self._frame[x_column].corr(self._frame[y_column])
        print(f"Correlation of {x_column} and {y_column} = {corr}")
        return  math.pow(corr,2)>= 0.7
    def plot_mutiply_regression(self, x_column: str, y_column: str, z_column: str,
                                title: str | None = None,
                                *, scatter_kwargs: dict | None = None,
                                line_kwargs: dict | None = None):
        """
        Perform and visualize multiple linear regression with 2 features.
        
        Args:
            x_column: First feature column name
            y_column: Second feature column name
            z_column: Target variable column name
            title: Plot title
            scatter_kwargs: Additional scatter plot options
            line_kwargs: Additional line plot options
        
        Returns:
            model: Trained LinearRegression model
        """

        
        data = self._frame[[x_column, y_column, z_column]].dropna()
        X = data[[x_column, y_column]].to_numpy()
        Y = data[z_column].to_numpy()
        
        if len(X) < 3:
            raise ValueError("Multiple regression requires at least 3 valid data points")
        
        model = linear_model.LinearRegression().fit(X, Y)
        r_sq = model.score(X, Y)
        
        print(f"Coefficient of determination (R-squared): {r_sq:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        print(f"Coefficients: {model.coef_}")
        print(f"  {x_column}: {model.coef_[0]:.4f}")
        print(f"  {y_column}: {model.coef_[1]:.4f}")
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter_opts = {"alpha": 0.6, "label": "observations", "s": 50, "c": "blue"}
        if scatter_kwargs:
            scatter_opts.update(scatter_kwargs)
        
        ax.scatter(X[:, 0], X[:, 1], Y, **scatter_opts)
        
        x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
        y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
        Z_mesh = model.predict(np.c_[X_mesh.ravel(), Y_mesh.ravel()]).reshape(X_mesh.shape)
            
        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.5, cmap='viridis')
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(z_column)
        ax.set_title(title or f"Multiple Regression: {z_column}")
        
        plt.tight_layout()
        return model
    def plot_correlation_of_all_dataset(self) -> None:
        """
        Create a correlation heatmap for all numeric columns.
        
        Displays a color-coded heatmap showing the correlation coefficients
        between all numeric columns in the dataset. Values range from -1 (perfect
        negative correlation) to +1 (perfect positive correlation).
        
        Args:
            None
        
        Returns:
            None (displays heatmap)
        """
        numeric_types = self._frame.select_dtypes(include=[np.number])
        correlation_matrix = numeric_types.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, fmt=".2f",
                    annot=True,
                    cmap="coolwarm",
                    linewidths=0.5)
        plt.title("All column correlations")
        plt.tight_layout()
        

        

    def __iter__(self) -> Iterator[pd.Series]:
        """
        Iterate over DataFrame rows.
        
        Args:
            None
        
        Returns:
            Iterator: Yields each row as a pandas Series
        """
        return (row for _, row in self._frame.iterrows())
    def graph_show(self):
        """
        Display all pending matplotlib plots.
        
        Args:
            None
        
        Returns:
            None (displays plots)
        """
        plt.show()
    def __len__(self) -> int:
        """
        Get the number of rows in the DataFrame.
        
        Args:
            None
        
        Returns:
            int: Number of rows
        """
        return len(self._frame)

    def sort(self, ByCategory: str, ascending: bool = True) -> pd.DataFrame:
        """
        Return a sorted copy of the DataFrame.
        
        Sorts the DataFrame by the specified column and returns a copy.
        The original DataFrame remains unchanged.
        
        Args:
            ByCategory: Column name to sort by
            ascending: Sort in ascending order if True (default: True);
                      descending order if False
        
        Returns:
            pd.DataFrame: Sorted DataFrame copy
        """
        return self._frame.sort_values(by=ByCategory, ascending=ascending).copy()
    
    
