import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted

# from matrepr import mdisplay


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred
        
        y_pred = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        y_pred = X @ self.weights_ #(NxD)(Dx1)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!
        
        w_opt = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        N = X.shape[0]
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0,0] = 0 #remove bias from being optimized
        
        w_opt =np.linalg.inv(X.T @ X + N*self.reg_lambda*I) @ X.T @ y #the solution from manually doing derivation
        
        #w_opt = np.linalg.solve(X.T @ X + self.reg_lambda*np.eye(n_features),  X.T @ y)
        #print(w_opt[..., np.newaxis].shape)
        # ========================
        #print(self.weights_)
        self.weights_ = w_opt#[..., np.newaxis]
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(model, df: DataFrame, target_name: str, feature_names: List[str] = None,):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    if feature_names is None: #if none we use alll cols - only remove the Y col
        feature_names = df.drop(columns=[target_name]).columns
    X = df[feature_names].to_numpy()
    y = df[target_name].to_numpy()
    #print(X.shape)
    # if not np.allclose(X[:,0], 1.0): #do bias trick - not sure how to use the BiasTrickTransfomer - if you even can
        # X = np.hstack([np.ones((X.shape[0],1)),X])
    #print(X.shape)
    y_pred = model.fit_predict(X,y)
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        #print(X.shape)
        ones = np.ones((X.shape[0],1)).astype(X.dtype)#(Dx1) 2x1
        #(X.dtype,ones.dtype)
        #print(ones.shape)
        
        xb = np.hstack((ones,X))
        assert np.all(xb[:,0] == 1) #check that the first col is indeed 1 only values
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        
        self.polynom = PolynomialFeatures(degree=self.degree,include_bias=True) #create a polynom fit function
        
        # ========================

    def fit(self, X, y=None):  #no neeed to touch      
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        # mdisplay(X, floatfmt=".2f")
        #raise NotImplementedError()
        #X-> 0=CRIM, 1=ZN, 2=INDUS, 3=CHAS, 4=NOX, 5=RM, 6=AGE, 7=DIS, 8=RAD, 9=TAX, 10=PTRATIO, 11=B, 12=LSTAT
        X_transformed = np.delete(X,4,axis=1) #remove chas
        # mdisplay(X_transformed, floatfmt=".2f")
        #Xt-> 0=CRIM, 1=ZN, 2=INDUS, 3=NOX, 4=RM, 5=AGE, 6=DIS, 7=RAD, 8=TAX, 9=PTRATIO, 10=B, 11=LSTAT
        X_transformed[:,1] = np.log1p(X_transformed[:,1]) #log(1+X) CRIM
        X_transformed[:,12] = np.log1p(X_transformed[:,12])#log(1+X) LSTAT
        # ========================
        #print(X.shape,X_transformed.shape)
        #X_trans = self.transform(X)
        X_transformed = self.polynom.fit_transform(X_transformed) #calculate what polymon fits the data after phi(x)=log(1+x)
        # mdisplay(X_transformed, floatfmt=".2f")
        # print("---")
        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features Both the returned sequences should be sorted so that the best (most correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    
    correlation_series = df.corr()[target_feature].drop(target_feature) #calc correlation for all features with target_feature and remove self-correlation because its not needed
    #print(correlation_series)
    top_correlation = correlation_series.abs().sort_values(ascending=False).head(n) #we want to sort but correlation is +-1 so we sort with abs and select the best N correlation, acending=false = Max-> min val sort
    #print(type(top_correlation))
    top_n_features = top_correlation.index.tolist() #convert df to list of names
    top_n_corr = correlation_series[top_n_features].tolist()  # save the top cor with their values +- and not just abs


    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Ground truth labels, shape (N,)
    :param y_pred: Predictions, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    mse = (np.square(y-y_pred)).mean()
    #print(mse)
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    y_bar = y.mean()
    #N = y.shape[0]
    #print(N)
    r2 = 1 - np.divide(mse_score(y,y_pred),mse_score(y, y_bar)) #no need to multiply by N, because we get N\N=1
    # ========================
    return r2


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds, degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities (recommended). See the docs for the sklearn.model_selection package http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree) you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has and their names. The parameters dict you return should use the same names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # def linreg_boston_kfold(model, x, y, fit=True):
    #     if fit:
    #         model.fit(x, y)
    #     y_pred = model.predict(x)
    #     mse = mse_score(y, y_pred)
    #     rsq = r2_score(y, y_pred)
    #     return y_pred, mse, rsq
    
    # kfold = sklearn.model_selection.KFold(n_splits=k_folds,shuffle=False)#,random_state=47) #create a splitter
    # print(f"K-Folders Splits: {kfold.get_n_splits(X)}")
    # best_mse, best_rsq = np.inf,-np.inf #init values
    
    # # og_model = copy.deepcopy(model)
    # #og_model = sklearn.base.clone(model)
    # #best_model = sklearn.base.clone(model)
    # for degree in degree_range:
    #     for _,reg_lambda in enumerate(lambda_range):
    #         #set the hyper paramters for this iteration
    #         # model.reg_lambda = reg_lambda
    #         # model.degree = degree
    #         #print(model.get_params())
    #         #model = sklearn.base.clone(og_model)
    #        # model.set_params()
    #         # Read current values
    #         #print(model.get_params()['bostonfeaturestransformer__degree'])
    #         #print(model.get_params()['linearregressor__reg_lambda'])
    #         #model = sklearn.base.clone(og_model)

    #         # Update hyperparameters
    #         model.set_params(bostonfeaturestransformer__degree=degree,linearregressor__reg_lambda=reg_lambda)
            
            
    #         mse_list = []
    #         rsq_list = []
            
    #         for i, (train_index, eval_index) in enumerate(kfold.split(X)):
                
    #             # print(f"Fold {i}:")
    #             # print(f"  Train: index={train_index}")
    #             # print(f"  Test:  index={test_index}")
    #             #First traim
    #             X_train = X[train_index,:]
    #             y_train = y[train_index]
                
                
                
    #             y_pred, mse_train, rsq_train = linreg_boston_kfold(model, X_train, y_train)
                
    #             #Then Check performance
    #             X_eval = X[eval_index,:]
    #             y_eval = y[eval_index]
    #             y_pred_eval, mse_eval, rsq_eval = linreg_boston_kfold(model, X_eval, y_eval,fit=False)
                
    #             #list so we can avg the kfolds
    #             mse_list.append(mse_eval)
    #             rsq_list.append(rsq_eval)
                
    #             #print data for debug
    #             # print(f"HyperP: lambda = {reg_lambda:.2f}, polydeg = {degree}\t| TestMSE [{mse_train:.2f}], EvalMSE [{mse_eval:.2f}]\t| TestR2 [{rsq_train:.2f}], EvalR2 [{rsq_eval:.2f}]")
    #             #Sava the best parameters if they are better mse\R2 than preious fold
    #         avg_mse = np.mean(mse_list)
    #         avg_rsq = np.mean(rsq_list)
            
    #         if avg_mse < best_mse or (avg_mse <= best_mse and avg_rsq > best_rsq):
    #             best_mse = avg_mse
    #             best_rsq = avg_rsq
    #             best_params = model.get_params()
    #             # best_model = sklearn.base.clone(model)

    #             #print("------")
    #             # print(f"Best Model: lambda = {reg_lambda:.2f}, polydeg = {degree}")
    #             # print(f"HyperP: lambda = {reg_lambda:.2f}, polydeg = {degree}\t| TrainMSE [{mse_train:.2f}], EvalMSE [{mse_eval:.2f}]\t| TrainR2 [{rsq_train:.2f}], EvalR2 [{rsq_eval:.2f}]")
                
    #             # print(model.named_steps['bostonfeaturestransformer'].degree)  
    #             # print(model.named_steps['linearregressor'].reg_lambda)        
    #             # print(best_params)
    #             # print("------")

                
            
    #         # print(X_train)
    #         # print(y_train)
            
    # # ========================

    # return best_params
    param_grid = {
        'bostonfeaturestransformer__degree': degree_range,
        'linearregressor__reg_lambda': lambda_range
    }

    grid_search = sklearn.model_selection.GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',   # Use R^2 to select the best hyperparameters
        cv=k_folds,
        n_jobs=-1   # Use all cpu cores
    )

    grid_search.fit(X, y)

    return grid_search.best_params_ #returns dict




