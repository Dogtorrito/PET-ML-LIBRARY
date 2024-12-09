from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

class LinearRegressor:
    """
    Class for implementing linear regression with several optimization methods.
     Available optimizers:
    - "MNK" (least squares),
    - "GD" (gradient descent),
    - "SGD" (stochastic gradient descent).
    
    Available loss functions:
    - "MSE" (Mean Squared Error),
    - "MAE" (Mean Absolute Error),
    - "MAPE" (Mean Absolute Percentage Error).

     Available regularizations:
    - "L1" (Lasso),
    - "L2" (Ridge).

    Example usage:
    --------------
    model = LinearRegressor(optim_type = "GD", n_iter = 1000, alpha = 0.01, loss = "MSE", regularization = "L1", lamb = "0.01")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    """
    def __init__(self,
                 optim_type: str,
                 n_iter: int = 0,
                 batch_size: int = 0,
                 regularization: str = '',
                 lamb: float = 0.0,
                 alpha: float = 0.0,
                 loss: str = ''):
        """
        A class constructor.
        
        :param optim_type: str
            Optimizer type ("MNK" - least squares, "GD" - gradient descent,
            "SGD" - stochastic gradient descent).
        :param n_iter: int
            Number of iterations (default 0).
        :param batch_size: int, optional
            Batch size for stochastic gradient descent (default 0).
        :param regularization: str, optional
            Type of regularization ("L1" - Lasso, "L2" - Ridge, '' - without)
            (default '').
        :param lamb: float, optional
            Regularization coefficient (default 0.0).
        :param alpha: float
            Learning rate (default 0.0).
        :param loss: str
            Loss function type: "MSE" - Mean Squared Error, "MAE" - Mean
            Absolute Error, "MAPE" - Mean Absolute Percentage Error
            (default '').
        """
        self.loss = loss
        self.alpha = alpha
        self.optim_type = optim_type
        self.regularization = regularization
        self.lamb = lamb
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.mape_iter = []
        self.mae_iter = []
        self.mse_iter = []
        self.w_iter = []
        
    def predict(self, X: np.ndarray):
        """
        Returns predictions for input data.

        :param X: np.ndarray
            Input Matrix
        :return: np.ndarray
            Vector of predictions
        """
        
        self.y_pred = X.dot(self.w)
        return self.y_pred

    def _regularization(self, w1: np.ndarray):
        """
        Calculating the model penalty when applying regularization.
        
        :param w1: np.ndarray
            Vector of weights.
        :return: np.ndarray
            Vector of penalties.
        """
        
        if self.regularization == 'L2':
            regular = 2*self.lamb*w1
            regular[0] = 0
        elif self.regularization == 'L1':
            regular = self.lamb*np.sign(w1)
            regular[0] = 0
        else:
            regular = 0
        return regular 

    def _calc_gradient(self, X: np.ndarray, y: np.ndarray, f: np.ndarray, w1: np.ndarray):
        """
        Calculating the Gradient for the Loss Function.

        :param X: np.ndarray
            Input matrix.
        :param y: np.ndarray
            Vector of target values. 
        :param f: np.ndarray
            Vector of predictions.
        :param w1: np.ndarray
            Vector of weights.
        :return: np.ndarray
            Gradient for updating weights.
        """
        
        N = len(X)
        if self.loss == "MSE":
            err = f - y
            grad = 2*np.dot(X.T, err)/N
        elif self.loss == "MAE":
            err = np.sign(f - y)
            grad = np.dot(X.T, err)/N
        elif self.loss == "MAPE":
            n = len(y)
            grad = np.zeros_like(w1)
            for i in range(n):
                if y[i] != 0:
                    error = (y[i] - f[i]) / y[i]
                    grad += (-100 / n) * np.sign(error) * X[i] / y[i]
        return grad
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains a linear regression model based on input data.

        :param X: np.ndarray
            The feature matrix for training data.
        :param y: np.ndarray
            The target values for training data.
        """
        
        size = X.shape[1]
        if self.optim_type == 'MNK':
            XtX = np.dot(X.T, X)
            XtXinv = np.linalg.pinv(XtX)
            XtXinv_Xt = np.dot(XtXinv, X.T)
            self.w = np.dot(XtXinv_Xt, y)
        elif self.optim_type == 'GD':
            w1 = np.random.normal(loc=1.0, scale=3.0, size=size)
            N = len(X)
            for _ in range(self.n_iter):
                f = np.dot(X, w1)
                regular = self._regularization(w1)
                grad = self._calc_gradient(X,y,f,w1) + regular
                w1 -= self.alpha*grad
            self.w = w1
        elif self.optim_type == 'SGD':
            w1 = np.random.normal(loc=1.0, scale=3.0, size=size)
            N = len(X)
            for _ in range(self.n_iter):
                indixes = np.random.permutation(N)
                X_shuffled=X[indixes]
                y_shuffled = y[indixes]
                for i in range(0,N,self.batch_size):   
                    random_index = np.random.randint(N)
                    X_batch = X_shuffled[i:i+self.batch_size]
                    y_batch = y_shuffled[i:i+self.batch_size]
                    f = np.dot(X_batch, w1)
                    regular = self._regularization(w1)
                    grad = self._calc_gradient(X_batch,y_batch,f,w1) + regular
                    w1 -= self.alpha*grad
            self.w = w1
#Для деревьев и MAPE MAE в линейной регрессии нормализация не нужна
#В деревьях не нужно смещение кажется (уточнить)

class Node:
    """
    A class that implements a node in a decision tree algorithm.
    """
    
    def __init__(self,
                 feature_idx: int = None,
                 threshold: float = None,
                 value: float = None,
                 left: Node = None,
                 right: Node = None):
        """
        Initializing a decision tree node.

        :param feature_idx: int, optional
            The index of the feature used for partitioning at this node
            (default None).
        :param threshold: float, optional
            Threshold for Data Splitting (default None).
        :param value: float, optional
            The value at the leaf node that represents the prediction
            (default None).
        :param left: Node, optional
            Left subtree containing nodes with features less than the threshold
            (default None).
        :param right: Node, optional
            The right subtree containing nodes with features greater than the threshold.
            (default None).
        """
        
        self.feature_idx = feature_idx    # Индекс признака для разделения
        self.threshold = threshold        # Порог разделения
        self.value = value                # Значение в листовом узле (среднее для регрессии)
        self.left = left                  # Левое поддерево (меньше порога)
        self.right = right                # Правое поддерево (больше порога)

class DecisionTreeRegressor:
    """
    A class to implement a decision tree used for regression.
    A decision tree is built by recursively dividing data based on features
    to minimize prediction error.

    Available criteria for evaluating split quality: "mse" (Mean Squared Error), "mae" (Mean Absolute Error)
    """
    
    def __init__(self, max_depth: int = None, criterion: str = 'mse'):
        """
        Initializing a decision tree model for regression.

        :param max_depth: int, optional
            Maximum tree depth. If None, the depth is unlimited (default = None).
        :param criterion: str, optional
            Criterion for assessing the quality of separations ('mse' or 'mae') (default = 'mse').
            Available criterions:
            - "mse" (Mean Squared Error),
            - "mae" (Mean Absolute Error).
        """
        
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains a decision tree model based on input data.

        :param X: np.ndarray
            The feature matrix for training data.
        :param y: np.ndarray
            The target values for training data.
        """
        
        self.root = self._build_tree(X, y, depth=0)

    def _calculate_criterion(self, y: np.ndarray):
        """
        Calculates the criterion value for the target y values.
        The criterion is used to assess the quality of data separation
        at each step of tree construction.

        :param y: np.ndarray
            Vector of target values.
        :return: float
            If 'mse' is selected, the variance of the target values ​​is returned.
            If 'mae' is selected, the average absolute deviation from the mean is returned.
        """
        
        if self.criterion == 'mse':
            return np.var(y)
        elif self.criterion == 'mae':
            return np.mean(np.abs(y - np.mean(y)))

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        """
        Recursively builds a decision tree based on data X and targets y.

        The algorithm at each depth level checks all possible separation thresholds for
        each feature and selects the best division that minimizes the selected
        error criterion (MSE or MAE). The process continues until the maximum depth is reached
        or until the error criterion becomes sufficiently small.

        :param X: np.ndarray
            Input matrix.
        :param y: np.ndarray
            Vector of target values.
        :param depth: int
        :return: Node
            A tree node containing either the threshold and feature index to split, or the value
            predictions in the case of a leaf node.
        """
        
        N, D = X.shape
        criterion_value = self._calculate_criterion(y)
        mean = np.mean(y)
        if depth == self.max_depth or criterion_value < 1e-5:
            return Node(value = mean)
        best_criterion = float('inf')
        best_idx, best_thresh = None, None
        for i in range(D):
            for threshold in np.unique(X[:, i]): 
                left_mask = X[:, i] <= threshold
                right_mask = X[:, i] > threshold
                left_criterion = self._calculate_criterion(y[left_mask])
                right_criterion = self._calculate_criterion(y[right_mask])
                weighted_criterion = (left_criterion * np.sum(left_mask) + right_criterion * np.sum(right_mask)) / N
                if weighted_criterion < best_criterion:
                    best_criterion = weighted_criterion
                    best_idx = i
                    best_thresh = threshold
        if best_idx is not None and best_thresh is not None:
            left_X, right_X = X[X[:, best_idx] <= best_thresh], X[X[:, best_idx] > best_thresh]
            left_y, right_y = y[X[:, best_idx] <= best_thresh], y[X[:, best_idx] > best_thresh]
            left_node = self._build_tree(left_X, left_y, depth + 1)
            right_node = self._build_tree(right_X, right_y, depth + 1)
            return Node(feature_idx=best_idx, threshold=best_thresh, left=left_node, right=right_node)
        else:
            return Node(value=np.mean(y))

    def predict(self, X: np.ndarray):
        """
        Predicts target values ​​for input data X based on the constructed tree.

        For each data instance, the recursive _predict_single method is called, which
        traverses the decision tree starting at the root node and returns the predicted value.

        :param X: np.ndarray
            Input matrix.
        :return: np.ndarray
            A vector of predicted values ​​for each data instance.
        """
        
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x: np.ndarray, node: Node):
        """
        Recursively predicts a value for a single data instance.

        For one data instance x, the method traverses the decision tree,
        starting from node node, and returns the predicted value based on
        thresholds and features contained in the tree nodes.

        :param x: np.ndarray
            One instance of input data
        :param node: Node
            The decision tree node at which recursion begins
            (usually the root node).
        :return: float
            The predicted value for this instance.
        """
        
        if node.left is None and node.right is None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)



class Random_Forest_Regression:
    """
    The Random_Forest_Regression class is an implementation of a random forest for solving regression problems,
    i.e. predicting a continuous value based on input data.
    Random forest is an ensemble method that combines multiple decision trees trained on random subsets of data
    and features. Averaging the predictions of all trees helps to obtain more accurate and robust results.
    """
    
    def __init__(self, n_trees: int, max_depth: int, criterion: str, random_state = None):
        """
        Initializes a random forest model for regression.

        :param n_trees: int
            Number of decision trees in the forest.
        :param max_depth: int or None, optional
            Maximum depth of each decision tree.
            If None, trees grow until all leaves are pure.
        :param criterion: str, optional
            The metric used to assess split quality in the trees. Options are:
            - "mse" (Mean Squared Error),
            - "mae" (Mean Absolute Error).
        :param random_state: int or None, optional
            Sets the seed for reproducibility.
        """
        self.n_trees = n_trees
        self.random_state = random_state
        self.max_depth = max_depth
        self.criterion = criterion
        self.trees = []
        self.bootstraps = []
        self.feature_indices = []

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """
        Creates a random bootstrap sample with replacement from the input data (X, y). 
        Each tree in the forest is trained on a unique sample to enhance model diversity.

        :param X: np.ndarray
            Matrix of test data.
        :param y: np.ndarray
            Vector of target values.
            
        :return: np.ndarray
            Return the bootstrap sample of X and corresponding labels y
            The samples are selected based on the randomly generated indices
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples: int, size = n_samples, replace = True)
        return X[indices], y[indices]

    def _random_feature_subset(self, n_features: int):
        """
        Selects random features for each bootstrap sample

        :param n_features: int
            Number of features in the dataset
        :return: np.ndarray
            Array containing the indices of the selected features
            which are used to create a unique feature subset for each decision tree
        """
        max_features = n_features//3
        Randindices = np.random.choice(n_features, max_features, replace=False)
        return Randindices

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains a random forest model based on input data.

        :param X: np.ndarray
            The feature matrix for training data.
        :param y: np.ndarray
            The target values for training data.
            Stores trained trees for use during prediction.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        np.random.seed(self.random_state)
        for _ in range(self.n_trees):
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            self.bootstraps.append((X_bootstrap, y_bootstrap))
            selected_features = self._random_feature_subset(n_features)
            self.feature_indices.append(selected_features)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,criterion=self.criterion)
            tree.fit(X_bootstrap[:, selected_features], y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X: np.ndarray):
         """
        Predicts target values for the test data based on the ensemble of trained models.

        :param X: np.ndarray
            The feature matrix for the test data.
        :return: np.ndarray
            A vector of predicted target values.
        """
        predictions = np.zeros((X.shape[0], self.n_trees), dtype=int)
        for i, tree in enumerate(self.trees):
            selected_features = self.feature_indices[i]
            predictions[:, i] = tree.predict(X[:, selected_features])
        final_predictions = np.mean(predictions,axis=1)
        return final_predictions

#Класс беггинга 
class Bagging_Forest_Regression:
    """
    A class for implementing Bagging (Bootstrap Aggregating) using decision trees for regression.

    Bagging is an ensemble learning technique where multiple decision trees are trained on 
    different bootstrap samples of the training data. The final prediction is the average 
    of the predictions of all trees.
    """
    
    def __init__(self, n_trees: int, depth: int, criterion: str, random_state: int = None):
        """
        Initializes the Bagging Forest regressor model with the specified parameters.
        :param n_trees: int
            The number of decision trees to train in the ensemble.
        :param depth: int
            The maximum depth for each decision tree.
        :param criterion: str
            The function used to measure the quality of
            a split in each tree ('mse' or 'mae').
        :param  random_state: int, optional (default = None)
            Seed for the random number generator to ensure reproducibility.
        """
        
        self.n_trees = n_trees
        self.depth = depth
        self.random_state = random_state
        self.criterion = criterion
        self.trees = []
        self.bootstraps = []

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """
        Creates a random bootstrap sample with replacement from the input data (X, y). 
        Each tree in the forest is trained on a unique sample to enhance model diversity.

        :param X: np.ndarray
            Matrix of test data.
        :param y: np.ndarray
            Vector of target values.
            
        :return: np.ndarray
            Return the bootstrap sample of X and corresponding labels y
            The samples are selected based on the randomly generated indices
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples: int, size = n_samples, replace = True)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains a bagging forest regression model based on input data.

        :param X: np.ndarray
            The feature matrix for training data.
        :param y: np.ndarray
            The target values for training data.
            Stores trained trees for use during prediction.
        """
        n_samples = X.shape[0]
        np.random.seed(self.random_state)
        for _ in range(self.n_trees):
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            self.bootstraps.append((X_bootstrap, y_bootstrap))
            tree = DecisionTreeRegressor(max_depth = self.depth,criterion = self.criterion)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X: np.ndarray):
        """
        Predicts target values for the test data based on the ensemble of trained models.

        :param X: np.ndarray
            The feature matrix for the test data.
        :return: np.ndarray
            A vector of predicted target values.
        """
        predictions = np.zeros((X.shape[0], self.n_trees), dtype = float)
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        final_predictions = np.mean(predictions,axis=1)
        return final_predictions

    bla1=(model,False,True)
    bla2=(model,False,True)
    bla3=(model,False,True)
    base_models  = [bla1,bla1,bla1]
    for model, flag1, flag2 in base_models 

class ModelWithConfig:
    """
    A class for storing a model and its configuration,
    including scaling parameters and adding intercepts.
    Used for stacking regressor.
    """
    def __init__(self,
                 model: yourmodel,
                 scaling_option: str = False,
                 intercept_option: bool = False
                ):
    """
    :param model: your model
        The model that will be trained
    :param scaling_option: str
        Option to scale the data before training the model.
        Possible values: 'MinMaxScaler', 'StandardScaler',
        'MaxAbsScaler', 'RobustScaler', or False (default, no scaling).
    :param intercept_option: bool
        Flag indicating whether to add an intercept column 
        (column of ones) to the data (default = False).
    """
        self.model = model
        self.scaling_option = scaling_option
        self.intercept_option = intercept_option

class StackingRegressor:
    """  
    For correct operation, it is necessary to wrap the models in the configurator class "class ModelWithConfig"
    A stacking regressor is an ensemble learning technique that combines the predictions of multiple base regression models
    to improve overall predictive   performance. In this approach, the base models are trained on the same dataset, 
    and their predictions are used as input features for a meta-model, which learns how to best combine them to produce the final output.

    For correct operation, it is necessary to wrap the models in the configurator class "class ModelWithConfig".

    """
    def __init__(self,
                 base_models: list[ModelWithConfig],
                 meta_model_with_config: ModelWithConfig,
                 val_size: float = 0.2,
                 random_state: int = None
                 ):
        """
        Initializes the Stacking regressor model.

        :param base_models: list[ModelWithConfig]
            List with basic models wrapped in a configurator class
        :param meta_model_with_config: ModelWithConfig
            Metamodel wrapped in a configurator class
        :param val_size: float
            Determines what portion of the data will be used
            to train the metamodel (default = 0.2)
        :param random_state: int
            Sets the initial state of the random number generator.
            This allows the random sampling process to be reproducible (default = None).
        """
        self.base_models = base_models
        self.meta_model_with_config = meta_model_with_config
        self.val_size = val_size
        self.random_state = random_state

        self.base_scalers = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'MaxAbsScaler': MaxAbsScaler(),
            'RobustScaler': RobustScaler()
        }

        self.meta_scalers = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'MaxAbsScaler': MaxAbsScaler(),
            'RobustScaler': RobustScaler()
        }

        self.scaler_trained = {key: False for key in self.base_scalers.keys()}
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains a Stacking Regressor model based on input data.

        :param X: np.ndarray
            The feature matrix for training data.
        :param y: np.ndarray
            The target values for training data.
        """
        self.reset_scalers()
        X_train, X_val, y_train, y_val = self.split_data(X, y)

        # Обучение базовых моделей
        for model_with_config in self.base_models:
            X_train_prepared = self.prepare_features(
                X_train,
                model_with_config.scaling_option,
                model_with_config.intercept_option,
                True,
                False
            )
            model_with_config.model.fit(X_train_prepared, y_train)

        # Подготовка данных для мета-модели
        val_predictions = []
        for model_with_config in self.base_models:
            X_val_prepared = self.prepare_features(
                X_val,
                model_with_config.scaling_option,
                model_with_config.intercept_option,
                False,
                False
            )
            val_predictions.append(model_with_config.model.predict(X_val_prepared))
            
        meta_features_val = np.column_stack(val_predictions)
        meta_features_val_prepared = self.prepare_features(
            meta_features_val,
            self.meta_model_with_config.scaling_option,
            self.meta_model_with_config.intercept_option,
            True,
            True
        )
        self.meta_model_with_config.model.fit(meta_features_val_prepared, y_val)
        

    def prepare_features(self,
                         X: np.ndarray,
                         scaling_option: str,
                         intercept_option: bool,
                         fit_scaler: bool = True,
                         use_meta_scalers: bool = False
                        ):
        """
        The function provides output scaling and/or adds interception.
        Possible scalers: MinMax Scaler, Standard Scaler,
        MaxAbs Scaler, Robust Scaler.

        You can preprocess the data in your own way,
        then preprocessing in the class configurator should be set to False.

        :param X: np.ndarray
            Input data
        :scaling option: str
            Option to scale the data before training the model.
            Possible values: 'MinMaxScaler', 'StandardScaler',
            'MaxAbsScaler', 'RobustScaler', or False.
        :intercept_option: bool 
            Flag indicating whether to add an intercept column 
            (column of ones) to the data (default = False).
        :fit_scaler: bool
            Flag that switches the scaler from training mode to data processing mode
            (default = True)
        :use_meta_scalers: bool 
            Flag toggling the function on preprocessing data for the metamodel
            (default = False)
        :return: X_prepared: np.ndarray 
            Return the prepeared data 
        """
        X_prepared = X.copy()

        scalers = self.meta_scalers if use_meta_scalers else self.base_scalers

        if scaling_option and scaling_option not in scalers:
            raise ValueError(f"Unknown scaling option: {scaling_option}")
        
        # Нормализация данных, если она указана для данной модели
        if scaling_option == 'MinMaxScaler':
            scaler_name = 'MinMaxScaler'
            scaler = scalers['MinMaxScaler']
        elif scaling_option == 'StandardScaler':
            scaler_name = 'StandardScaler'
            scaler = scalers['StandardScaler']
        elif scaling_option == 'MaxAbsScaler':
            scaler_name = 'MaxAbsScaler'
            scaler = scalers['MaxAbsScaler']
        elif scaling_option == 'RobustScaler':
            scaler_name = 'RobustScaler'
            scaler = scalers['RobustScaler']
        else:
            scaler_name = None
            scaler = None

        if scaler:
            if fit_scaler and not use_meta_scalers:
                if not self.scaler_trained[scaler_name]:
                    X_prepared = scaler.fit_transform(X_prepared)
                    self.scaler_trained[scaler_name] = True
            elif fit_scaler:
                X_prepared = scaler.fit_transform(X_prepared)
            else:
                X_prepared = scaler.transform(X_prepared)
            
        if intercept_option:
            X_prepared = np.hstack((np.ones((X_prepared.shape[0], 1)), X_prepared))
        
        return X_prepared

    def split_data(self, X: np.ndarray, y: np.ndarray):
        """
        The function divides the input data into 2 parts: training,
        on which the base models are trained, and validation. On the validation,
        predictions of the base models are made, on these predictions the metamodel is trained.

        :param X: np.ndarray
            The feature matrix for training data.
        :param y: np.ndarray
            The target values for training data.

        :return X_train: np.ndarray
            The feature matrix for training base models
        :return X_val: np.ndarray
            The feature matrix for trainig meta model
        :return y_train: np.ndarray
            The target values for training base models
        :return y_val: np.ndarray
            The target values for training meta model
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = self.val_size, random_state = self.random_state)

        return X_train, X_val, y_train, y_val

    def predict(self, X: np.ndarray):
        """
        :param X: np.ndarray
            The matrix of features on which predictions will be made.
        :return: np.ndarray
            A vector of predicted target values.
        """
        # Получение предсказаний базовых моделей
        test_predictions = []
        for model_with_config in self.base_models:
            X_test_prepared = self.prepare_features(
                X,
                model_with_config.scaling_option,
                model_with_config.intercept_option,
                False,
                False
            )
            test_predictions.append(model_with_config.model.predict(X_test_prepared))
        meta_features_test = np.column_stack(test_predictions)
        meta_features_test_prepared = self.prepare_features(
            meta_features_test,
            self.meta_model_with_config.scaling_option,
            self.meta_model_with_config.intercept_option,
            False,
            True
            )
        return self.meta_model_with_config.model.predict(meta_features_test_prepared)

    def reset_scalers(self):
        """
        Resets the state of the scalers to ensure
        that the model is trained correctly with new data.
        """
        
        self.scaler_trained = {key: False for key in self.base_scalers}
        
        self.base_scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler()
    }

        self.meta_scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler()
    }
        

class Gradient_Boosting_Regressor:
    """
    A class for implementing gradient boosting for regression tasks.

    Gradient boosting is an ensemble technique that combines multiple weak models
    (typically decision trees) to form a strong predictive model. Each model is trained
    to correct the residual errors of the previous models.
    """
    
    def __init__(self, max_depth: int = 1, criterion: str = 'mse'):
        """
        Initializes the gradient boosting regressor with specified parameters.

        :param max_depth: int, optional (default=1)
            The maximum depth for each decision tree.
        :param criterion: str, optional (default='mse')
            The criterion used to measure the quality of a split ('mse' or 'mae').
        """
        
        self.max_depth = max_depth
        self.criterion = criterion
        

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_models: int = 10, lr: float = 0.1):
        """
        Trains the gradient boosting model by sequentially fitting decision trees to the residuals.

        :param X_train: np.ndarray
            The feature matrix for training data.
        :param y_train : np.ndarray
            The target values for training data.
        :param n_models : int, optional (default=10)
            The number of decision trees to train.
        :param lr: float, optional (default=0.1)
            Learning rate that controls how much each tree contributes to the final model. 
        """
        
        self.models = []
        self.lr = lr
        self.initial_prediction = np.mean(y_train)
        predictions = np.full(y_train.shape, self.initial_prediction)
        
        for i in range(n_models):
            residuals = y_train - predictions
            model = DecisionTreeRegressor(max_depth = self.max_depth, criterion = self.criterion)
            model.fit(X_train, residuals)
            self.models.append(model)
            new_predictions = model.predict(X_train)
            predictions += self.lr * new_predictions        
        
    def predict(self, X_test: np.ndarray):
        """
        Predicts target values for the test data based on the ensemble of trained models.

        :param X_test: np.ndarray
            The feature matrix for the test data.
        :return: np.ndarray
            A vector of predicted target values.
        """
        
        test_predictions = np.full(X_test.shape[0], self.initial_prediction)
        for model in self.models:
            test_predictions += self.lr * model.predict(X_test)
        self.y_pred = test_predictions
        return test_predictions

    def score(self, y_test: np.ndarray):
        """
        Evaluates the model performance using MSE and MAPE.

        :param y_test: np.ndarray
            The true target values for the test data.
        """
        
        print('MSE')
        print(mean_squared_error(y_test, self.y_pred))
        print('MAPE')
        print(mean_absolute_percentage_error(y_test, self.y_pred))

class LogitRegressor:
    """
    A class for implementing logistic regression using gradient descent.
    Logistic regression is used to predict the probability of belonging to
    one of two classes. The model is optimized using gradient descent,
    minimizing the log loss function.
    """
    
    def __init__(self, alpha: float):
        """
        Initialize a logistic regression model with a given learning rate.

        :param alpha: float
            Learning rate 
        """
        self.alpha = alpha
        self.LOGloss_iter = []

    def sigmoid(self,z: np.ndarray):
        """
        Computes the sigmoid function for input z values.

        :param z: np.ndarray
            Input values ​​for which the sigmoid is calculated
        :return: np.ndarray
            The sigmoid function values ​​for each element in z.
        """
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
        
    def predict_proba(self, X: np.ndarray):
        """
        Returns the predicted probabilities of belonging to class 1 for each sample.

        :param X: np.ndarray
            Matrix of features for prediction
        :return: np.ndarray
            Probabilities of belonging to class 1 for each sample.
        """
        self.y_pred = self.sigmoid(np.dot(X, self.w))
        return self.y_pred

    def predict(self,X: np.ndarray):
        """
        Returns the predicted class labels (0 or 1) based on a threshold of 0.5.

        :param X: np.ndarray
            Matrix of features for prediction
        :return: np.ndarray
            Vector of predicted class labels (0 or 1).
        """
        pred_proba = sigmoid(np.dot(X, self.w))
        pred = (pred_proba >= 0.5).astype(int)
        return pred
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_iter: int):
        """
        Trains a logistic regression model using gradient descent.

        :param X: np.ndarray
            Input matrix for training
        :param y: np.ndarray
            Vector of target values
        """
        size = X.shape[1]
        np.random.seed(42)
        w1 = np.random.normal(loc=1.0, scale=3.0, size=size)
        N = len(X)
        for _ in range(n_iter):
            f = sigmoid(np.dot(X, w1))
            err = y - f
            grad = -np.dot(X.T, err)/N
            w1 -= self.alpha*grad
            fnew = sigmoid(np.dot(X, w1))
            self.LOGloss_iter.append(log_loss(y,fnew))
        self.w = w1