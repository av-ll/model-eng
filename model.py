from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV,cross_validate
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import pickle
import numpy as np
import scipy.stats as st

warnings.simplefilter('ignore')

class Model:
    
    """Takes 3 parameters to initiate name - the name of the model,
    
    model - the model to use and params - the parameter grid for RandomizedSearchCV"""
    
    def __init__(self,name,model,param_grid):
        
        self.name = name
        
        self.model = model
        
        self.param_grid = param_grid
         
    def random_search(self, X, y, n_iter=150, cv=8, scoring=['f1','accuracy'], random_state=0, use_scaling=False):
        
        """
        Perform random search cross-validation for hyperparameter tuning.
        
        Parameters are the same as for the libraries used
        
        """
        
        # initialize StratifiedKFold cross-validator
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # initialize scaler and estimator for the pipeline
        scaler = RobustScaler()
        estimator = self.model
    
        if use_scaling:
            # If use_scaling is True, use the scaler and estimator within a pipeline
            pipeline = Pipeline([('scaler',scaler),('model',self.model)])
            param_grid = {'model__' + k: v for k, v in self.param_grid.items()}  # Add 'model__' prefix to keys
        else:
            # If use_scaling is False, use only the estimator
            pipeline = estimator
            param_grid = self.param_grid
        
        # initialize randomized search cross-validator
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            refit='f1',
            n_iter=n_iter,
            cv=skf,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=True
        )
        
        # perform randomized search on data
        random_search.fit(X, y)
        if use_scaling:
            self.best_model_scaled = random_search.best_estimator_[1]
        else:# return best model with hyperparameters
            self.best_model = random_search.best_estimator_
        
        
        
        
        
    def cv_model_data(self,X,y,cv=8,scoring=['f1','accuracy'],random_state=0):
        
        """
       Perform cross-validation on the given data using the best model obtained after RandomizedSearchCV
       
       Calculates the train and test scores means
    for both accuracy and f1 score.
       
       """
        
       # initialize StratifiedKFold cross-validator
        skf = StratifiedKFold(n_splits=cv,random_state=random_state,shuffle=True)
        
        # cross-validate the best model on the data
        
        model_skf_score = cross_validate(self.best_model,X,y,scoring=scoring,cv=skf,n_jobs=-1,return_train_score=True)
        
        # calculate the mean scores for train and test data
        
        self.train_f1_score = model_skf_score['train_f1'].mean().round(4)
        
        self.train_accuracy_score = model_skf_score['train_accuracy'].mean()
        
        self.test_f1_score = model_skf_score['test_f1'].mean().round(4)
        
        self.test_accuracy_score = model_skf_score['test_accuracy'].mean()
        
        self.test_f1_score_stdev = np.std(model_skf_score['test_f1'])
        
        self.test_accuracy_score_stdev = np.std(model_skf_score['test_accuracy'])
        
        self.confidence_interval_accuracy = st.t.interval(alpha=0.95,
                                                 df=len(model_skf_score['test_accuracy'])-1,
                                                 loc=np.mean(model_skf_score['test_accuracy']),
                                                 scale=st.sem(model_skf_score['test_accuracy'])
                                                 )
        self.confidence_interval_f1 = st.t.interval(alpha=0.95,
                                                 df=len(model_skf_score['test_f1'])-1,
                                                 loc=np.mean(model_skf_score['test_f1']),
                                                 scale=st.sem(model_skf_score['test_f1'])
                                                 )
        
        self.confidence_interval_accuracy = tuple(round(val*100, 4) for val in self.confidence_interval_accuracy)
        
        self.confidence_interval_f1 = tuple(round(val, 4) for val in self.confidence_interval_f1)
        
        # print the train and test scores for the model
        
        print(f'\n{self.name} train accuracy score mean: ', (self.train_accuracy_score*100).round(4),'%')

        print(f'\n{self.name} test accuracy score mean: ',(self.test_accuracy_score*100).round(4),'%')

        print(f'\n{self.name} train f1 score mean: ', self.train_f1_score)

        print(f'\n{self.name} test f1 score mean: ', self.test_f1_score)
        
        print(f'\n{self.name} accuracy score confidence interval: ',self.confidence_interval_accuracy)
        
        print(f'\n{self.name} f1 score confidence interval: ', self.confidence_interval_f1,'\n')
        
        
    def cv_model_scaled_data(self,X,y,cv=8,n_iter=150,scoring=['f1','accuracy'],random_state=0):
        
        """
        Perform cross-validation on the best estimator with scaled data.
        
        """
        
        # initialize StratifiedKFold cross-validator
        
        skf = StratifiedKFold(n_splits=cv,random_state=random_state,shuffle=True)
        
        # initialize a pipeline with scaler and best estimator
        pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', self.best_model_scaled)
])
        
        # perform cross-validation on scaled data scaler is added to the pipeline
        # in order to prevent data leakage
        scaled_data_scores = cross_validate(pipeline, X, y, cv=skf,scoring=scoring,n_jobs=-1,return_train_score=True)
        
        """  Prints:
        - The train accuracy score mean for the model with scaled data.
        - The test accuracy score mean for the model with scaled data.
        - The train f1 score mean for the model with scaled data.
        - The test f1 score mean for the model with scaled data. """
        
        self.train_f1_score_scaled = scaled_data_scores['train_f1'].mean().round(4)
        
        self.train_accuracy_score_scaled = scaled_data_scores['train_accuracy'].mean()
        
        self.test_f1_score_scaled = scaled_data_scores['test_f1'].mean().round(4)
        
        self.test_accuracy_score_scaled = scaled_data_scores['test_accuracy'].mean()
        
        self.confidence_interval_accuracy_scaled = st.t.interval(alpha=0.95,
                                                 df=len(scaled_data_scores['test_accuracy'])-1,
                                                 loc=np.mean(scaled_data_scores['test_accuracy']),
                                                 scale=st.sem(scaled_data_scores['test_accuracy'])
                                                 )
        self.confidence_interval_f1_scaled = st.t.interval(alpha=0.95,
                                                 df=len(scaled_data_scores['test_f1'])-1,
                                                 loc=np.mean(scaled_data_scores['test_f1']),
                                                 scale=st.sem(scaled_data_scores['test_f1'])
                                                 )
        
        self.confidence_interval_accuracy_scaled = tuple(round(val*100, 4) for val in self.confidence_interval_accuracy_scaled)
        
        self.confidence_interval_f1_scaled = tuple(round(val, 4) for val in self.confidence_interval_f1_scaled)
        
        print(f'\n{self.name} train accuracy score mean scaled data: ', (self.train_accuracy_score_scaled*100).round(4),'%')

        print(f'\n{self.name} test accuracy score mean scaled data: ',(self.test_accuracy_score_scaled*100).round(4),'%')

        print(f'\n{self.name} train f1 score mean scaled data: ', self.train_f1_score_scaled)

        print(f'\n{self.name} test f1 score mean scaled data: ', self.test_f1_score_scaled)
        
        print(f'\n{self.name} accuracy score confidence interval scaled: ', self.confidence_interval_accuracy_scaled)
        
        print(f'\n{self.name} f1 score confidence interval scaled: ', self.confidence_interval_f1_scaled)
        
    
    def save_model(self, filepath, scaled_model = True):
        
        """
        Save the best model to a file.
        
        You can specify which model to save 
        
        the model for the scaled data or the model
        
        for the base data
        
        scaled_data = True for scaled and = False for base data"""
        
        if scaled_model:
            
            with open(filepath, 'wb') as f:
            
                pickle.dump(self.best_model_scaled, f)
                
        elif not scaled_model:
            
            with open(filepath, 'wb') as f:
            
                pickle.dump(self.best_model, f)
        
        
        
