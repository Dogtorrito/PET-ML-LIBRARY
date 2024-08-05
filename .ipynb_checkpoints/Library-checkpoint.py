from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
#Линейный регрессор
class LinearRegressor:
    def __init__(self,optim_type,n_iter=0,batch_size = 0,regularization='',lamb=0,alpha=0,loss=''):
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
    def predict(self, X):
        self.y_pred = X.dot(self.w)
        return self.y_pred

    def _regularization(self,w1):
        if self.regularization == 'L2':
            regular = 2*self.lamb*w1
            regular[0] = 0
        elif self.regularization == 'L1':
            regular = self.lamb*np.sign(w1)
            regular[0] = 0
        else:
            regular = 0
        return regular 

    #def _metrics(self, X, y, w1):
        #self.w_iter.append(copy.copy(w1))
        #self.mae_iter.append(mean_absolute_error(y, np.dot(X, w1)))
        #self.mse_iter.append(mean_squared_error(y, np.dot(X, w1)))
        #self.mape_iter.append(mean_absolute_percentage_error(y, np.dot(X, w1)))
    
    def _calc_gradient(self,X,y,f,w1):
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
    def fit(self, X, y):
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
                #self._metrics(X, y, w1)
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
                #self._metrics(X, y, w1)
            self.w = w1

#дерево решающее регрессор 
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import copy
#Для деревьев и MAPE MAE в линейной регрессии нормализация не нужна
#В деревьях не нужно смещение кажется (уточнить)

class Node:
    def __init__(self, feature_idx=None , threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx    # Индекс признака для разделения
        self.threshold = threshold        # Порог разделения
        self.value = value                # Значение в листовом узле (среднее для регрессии)
        self.left = left                  # Левое поддерево (меньше порога)
        self.right = right                # Правое поддерево (больше порога)

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, criterion='mse'):
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _calculate_criterion(self, y):
        if self.criterion == 'mse':
            return np.var(y)
        elif self.criterion == 'mae':
            return np.mean(np.abs(y - np.mean(y)))

    def _build_tree(self, X, y, depth):
        N, D = X.shape
        criterion_value = self._calculate_criterion(y)
        mean = np.mean(y)

        if depth == self.max_depth or criterion_value < 1e-5:
            return Node(value=mean)

        best_criterion = float('inf')
        best_idx, best_thresh = None, None

        for i in range(D): #Ищем оптимальные критерии деления датасета на 2 группы.
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
            return Node(value=np.mean(y))  # возврат листового узла среднего значения

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.left is None and node.right is None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

class Random_Forest_Regression:
    def __init__(self,n_trees,max_depth,criterion,random_state=None):
        self.n_trees = n_trees
        self.random_state = random_state
        self.max_depth = max_depth
        self.criterion = criterion
        self.trees = []
        self.bootstraps = []
        self.feature_indices = []  # Для хранения индексов выбранных признаков

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _random_feature_subset(self,n_features):
        max_features = n_features//3
        Randindices = np.random.choice(n_features, max_features, replace=False)
        return Randindices

    def fit(self, X, y):
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
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_trees), dtype=int)
        for i, tree in enumerate(self.trees):
            selected_features = self.feature_indices[i]
            predictions[:, i] = tree.predict(X[:, selected_features])

        final_predictions = np.mean(predictions,axis=1)
        return final_predictions

#Класс беггинга 
class Bagging_Forest_Regression:
    def __init__(self,n_trees,depth,criterion,random_state=None):
        self.n_trees = n_trees
        self.depth = depth
        self.random_state = random_state
        self.criterion = criterion
        self.trees = []
        self.bootstraps = []
        #self.feature_indices = []  # Для хранения индексов выбранных признаков

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        n_samples = X.shape[0]
        #n_features = X.shape[1]
        np.random.seed(self.random_state)
        for _ in range(self.n_trees):
            # Создание бутстрап-выборки
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            self.bootstraps.append((X_bootstrap, y_bootstrap))
            
            tree = DecisionTreeRegressor(max_depth=self.depth,criterion=self.criterion)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_trees), dtype=float)
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)

        final_predictions = np.mean(predictions,axis=1)
        return final_predictions

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class StackingRegressor:
    def __init__(self, base_models, meta_model,file_path,test_size,val_size,random_state=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.file_path=file_path

    def fit(self):
        # Получение данных из внешней функции Data loader
        data = Data_loader(file_path=self.file_path, test_size=self.test_size,val_size=self.val_size, random_state=self.random_state)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.X_train_NO = data['X_train_NO']
        self.X_test_NO = data['X_test_NO']
        self.X_train_interc = data['X_train_interc']
        self.X_test_interc = data['X_test_interc']
        self.X_train_NO_interc = data['X_train_NO_interc']
        self.X_test_NO_interc = data['X_test_NO_interc']
        self.X_val=data['X_val']
        self.X_val_interc = data['X_val_interc']
        self.X_val_NO_interc = data['X_val_NO_interc']
        self.X_val_NO = data['X_val_NO']
        self.y_val = data['y_val']
        
        
        
        # Обучение базовых моделей на обучающей выборке 
        for model in self.base_models:
            features = self.get_features_from_train(model)
            model.fit(features, self.y_train)
        
        # Предсказания базовых моделей на валидацинной выборке для обучения мета-модели
        predictions_val = []
        for model in self.base_models:
            features_val=self.get_features_from_val(model)
            predictions_val.append(model.predict(features_val))
        #Передача предсказаний как фичи для мета-модели
        meta_features_val = np.column_stack(predictions_val)
        # Обучение мета-модели на предсказаниях базовых моделей
        self.meta_model.fit(meta_features_val, self.y_val)

    def get_features_from_train(self,model):
        if isinstance(model, LinearRegressor):
            if model.loss == 'MSE' or model.optim_type == 'MNK':
                return self.X_train_interc #вроде правильно 
            elif model.loss == 'MAE' or model.loss == 'MAPE':
                return self.X_train_NO_interc #
        #elif isinstance(model, MNK):
            #return self.X_train
        else:
            return self.X_train_NO
            
    def get_features_from_val(self,model):
        if isinstance(model, LinearRegressor):
            if model.loss == 'MSE' or model.optim_type == 'MNK':
                return self.X_val_interc
            elif model.loss == 'MAE' or model.loss == 'MAPE':
                return self.X_val_NO_interc
        #elif isinstance(model, MNK):
            #return self.X_val
        else:
            return self.X_val_NO

    def get_features_from_test(self,model):
        if isinstance(model, LinearRegressor):
            if model.loss == 'MSE' or model.optim_type == 'MNK':
                return self.X_test_interc
            elif model.loss == 'MAE' or model.loss == 'MAPE':
                return self.X_test_NO_interc
        #elif isinstance(model, MNK):
            #return self.X_test
        else:
            return self.X_test_NO

    def predict(self):
        # Предсказания базовых моделей на тестовой выборке 
        predictions_test = []
        for model in self.base_models:
            features_test=self.get_features_from_test(model)
            predictions_test.append(model.predict(features_test))
        #Передача предсказаний как фичи для мета-модели
        meta_features_test = np.column_stack(predictions_test)
        # Предсказание мета-модели на основе предсказаний базовых моделей на тестовой выборке 
        return self.meta_model.predict(meta_features_test)

# функция загрузки данных для удобства. нуэно его дописать и внедрить в сложные модели
def Data_loader(file_path,test_size=0.3,val_size=0.5,random_state=42):
    df = pd.read_csv(file_path)
    # Перемешивание данных
    df = df.sample(frac=1).reset_index(drop=True)
    train_df, other_df = train_test_split(df, test_size=test_size, random_state=random_state)  # используем random_state для воспроизводимости
    val_df, test_df = train_test_split(other_df, test_size=val_size, random_state=random_state)
    # Проверяем размеры полученных выборок
    print("Размер обучающей выборки:", train_df.shape)
    print("Размер тестовой выборки:", test_df.shape)
    #train_df.to_csv('train_datasetLN3.csv', index=False)
    #test_df.to_csv('test_datasetLN3.csv', index=False)
    #TRAIN = pd.read_csv('train_datasetLN3.csv')
    #TEST = pd.read_csv('test_datasetLN3.csv')
    TRAIN = train_df.copy()
    TEST = test_df.copy()
    TRAINscaller = train_df.copy()
    TESTscaller = test_df.copy()
    TRAINnoSc = train_df.copy()
    TESTnoSc = test_df.copy()
    VALscaller = val_df.copy()
    VALnoSc = val_df.copy()
    # Выбираем столбцы для нормализации
    columns_to_normalize = ['LONGITUDE', 'LATITUDE', 'BATH','BEDS','PROPERTYSQFT']
    # Применяем Min-Max Scaling
    TRAINscaller[columns_to_normalize] = TRAINscaller[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # Скеллинг тестового датасета
    TESTscaller[columns_to_normalize] = TESTscaller[columns_to_normalize].apply(lambda x: (x - TRAIN[x.name].min()) / (TRAIN[x.name].max() - TRAIN[x.name].min()))
    VALscaller[columns_to_normalize] = VALscaller[columns_to_normalize].apply(lambda x: (x - TRAIN[x.name].min()) / (TRAIN[x.name].max() - TRAIN[x.name].min()))
    #Создаем новые датасеты
    X_train = TRAINscaller[['LONGITUDE', 'LATITUDE','BATH','BEDS','PROPERTYSQFT']].to_numpy()
    y_train = TRAINscaller['PRICE'].to_numpy()
    X_test = TESTscaller[['LONGITUDE', 'LATITUDE','BATH','BEDS','PROPERTYSQFT']].to_numpy()
    y_test = TESTscaller['PRICE'].to_numpy()
    X_train_NO = TRAINnoSc[['LONGITUDE', 'LATITUDE','BATH','BEDS','PROPERTYSQFT']].to_numpy()
    X_test_NO = TESTnoSc[['LONGITUDE', 'LATITUDE','BATH','BEDS','PROPERTYSQFT']].to_numpy()
    X_val = VALscaller[['LONGITUDE', 'LATITUDE','BATH','BEDS','PROPERTYSQFT']].to_numpy()
    y_val = VALscaller['PRICE'].to_numpy()
    X_val_NO = VALnoSc[['LONGITUDE', 'LATITUDE','BATH','BEDS','PROPERTYSQFT']].to_numpy()
    X_train_interc = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_interc = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    X_train_NO_interc = np.hstack((np.ones((X_train_NO.shape[0], 1)), X_train_NO))
    X_test_NO_interc = np.hstack((np.ones((X_test_NO.shape[0], 1)), X_test_NO))
    X_val_interc = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
    X_val_NO_interc = np.hstack((np.ones((X_val_NO.shape[0], 1)), X_val_NO))
    #return X_train,y_train,X_test,y_test,X_train_NO,X_test_NO,X_train_interc,X_test_interc, X_train_NO_interc,X_test_NO_interc
    return {
        'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'X_train_NO': X_train_NO,'X_test_NO':X_test_NO,
        'X_train_interc': X_train_interc, 'X_test_interc': X_test_interc, 'X_train_NO_interc': X_train_NO_interc,'X_test_NO_interc':X_test_NO_interc,
        'X_val_interc':X_val_interc,'X_val_NO_interc':X_val_NO_interc,'y_val':y_val,'X_val_NO':X_val_NO,'X_val':X_val
    }

class Gradient_Boosting_Regressor:
    def __init__(self,max_depth = 1,criterion = 'mse'):
        self.max_depth = max_depth
        self.criterion = criterion
        pass

    def fit(self,X_train,y_train,n_models = 10,lr = 0.1):
        self.models = []
        self.lr = lr
        self.initial_prediction = np.mean(y_train)
        predictions = np.full(y_train.shape, self.initial_prediction)
        
        for i in range(n_models):
            # Вычисление остатков
            residuals = y_train - predictions
    
            # Обучение новой модели на остатках
            model = DecisionTreeRegressor(max_depth=self.max_depth,criterion=self.criterion)
            model.fit(X_train, residuals)
    
            # Сохранение модели
            self.models.append(model)
    
            # Новое предсказание на обучающих данных
            new_predictions = model.predict(X_train)
            predictions += self.lr * new_predictions        
        

    def predict(self,X_test):
        test_predictions = np.full(X_test.shape[0], self.initial_prediction)
        for model in self.models:
            test_predictions += self.lr * model.predict(X_test)
        self.y_pred = test_predictions
        return test_predictions

    def score(self,y_test):
        print('MSE')
        print(mean_squared_error(y_test, self.y_pred))
        print('MAPE')
        print(mean_absolute_percentage_error(y_test, self.y_pred))

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import copy

class LogitRegressor:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.LOGloss_iter = []

    def sigmoid(self,z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
        
    def predict_proba(self, X):
        self.y_pred = self.sigmoid(np.dot(X, self.w))
        return self.y_pred

    def predict(self,X):
        pred_proba = sigmoid(np.dot(X, self.w))
        pred = (pred_proba >= 0.5).astype(int)
        return pred
    
    def fit(self, X, y, n_iter):
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
            #print(log_loss(y,fnew))
        self.w = w1