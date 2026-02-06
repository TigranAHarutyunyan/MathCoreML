class CleanData:
    def __init__(self, x_q1, x_q3, y_q1, y_q3):
        """
        Initialize CleanData with quartile values for outlier detection.
        
        Args:
            x_q1: First quartile (25th percentile) value for X
            x_q3: Third quartile (75th percentile) value for X
            y_q1: First quartile (25th percentile) value for Y
            y_q3: Third quartile (75th percentile) value for Y
        
        Returns:
            None
        """
        # Store X quartile values for X dimension outlier detection
        self._x_q1 = x_q1
        self._x_q3 = x_q3
        # Store Y quartile values for Y dimension outlier detection
        self._y_q1 = y_q1
        self._y_q3 = y_q3
    @property
    def x_iqr(self) -> float:
        """
        Get the Interquartile Range (IQR) of X.
        
        IQR = Q3 - Q1, represents the middle 50% of data spread.
        Used in the IQR method for outlier detection.
        
        Returns:
            float: The IQR value (Q3 - Q1)
        """
        return self._x_q3 - self._x_q1
    @property
    def y_iqr(self) -> float:
        """
        Get the Interquartile Range (IQR) of Y.
        
        IQR = Q3 - Q1, represents the middle 50% of data spread.
        Used in the IQR method for outlier detection.
        
        Returns:
            float: The IQR value (Q3 - Q1)
        """
        return self._y_q3 - self._y_q1
    @property
    def x_lower_bound(self) -> float:
        """
        Get the lower bound for X outlier detection.
        
        Uses IQR method: Lower Bound = Q1 - 1.5 * IQR
        Values below this bound are considered outliers.
        
        Returns:
            float: Lower bound calculated as Q1 - 1.5 * IQR
        """
        return self._x_q1 - 1.5 * self.x_iqr
    @property
    def x_upper_bound(self) -> float:
        """
        Get the upper bound for X outlier detection.
        
        Uses IQR method: Upper Bound = Q3 + 1.5 * IQR
        Values above this bound are considered outliers.
        
        Returns:
            float: Upper bound calculated as Q3 + 1.5 * IQR
        """
        return self._x_q3 + 1.5 * self.x_iqr
    @property
    def y_lower_bound(self) -> float:
        """
        Get the lower bound for Y outlier detection.
        
        Uses IQR method: Lower Bound = Q1 - 1.5 * IQR
        Values below this bound are considered outliers.
        
        Returns:
            float: Lower bound calculated as Q1 - 1.5 * IQR
        """
        return self._y_q1 - 1.5 * self.y_iqr
    @property
    def y_upper_bound(self) -> float:
        """
        Get the upper bound for Y outlier detection.
        
        Uses IQR method: Upper Bound = Q3 + 1.5 * IQR
        Values above this bound are considered outliers.
        
        Returns:
            float: Upper bound calculated as Q3 + 1.5 * IQR
        """
        return self._y_q3 + 1.5 * self.y_iqr
    def y_is_outlier(self, value) -> bool:
        """
        Check if a Y value is an outlier using the IQR method.
        
        An outlier is any value that falls outside the bounds:
        (value < lower_bound) or (value > upper_bound)
        
        Args:
            value: The Y value to check
        
        Returns:
            bool: True if value is outside the bounds (outlier), False otherwise
        """
        return (value < self.y_lower_bound) or (value > self.y_upper_bound)
    def x_is_outlier(self, value) -> bool:
        """
        Check if an X value is an outlier using the IQR method.
        
        An outlier is any value that falls outside the bounds:
        (value < lower_bound) or (value > upper_bound)
        
        Args:
            value: The X value to check
        
        Returns:
            bool: True if value is outside the bounds (outlier), False otherwise
        """
        return (value < self.x_lower_bound) or (value > self.x_upper_bound)
        
        