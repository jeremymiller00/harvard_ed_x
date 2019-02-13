
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class ScaleSubset(BaseEstimator, TransformerMixin):
    '''
    Use sklearn StandardScalar to scale only numeric columns.

    Parameters:
    ----------
    input {dataframe, list}: dataframe containing mixed featurevariable    types, list of names of numeric feature columns
    output: {dataframe}: dataframe with numeric features scaledand     categorical features unchanged

    '''
    def __init__(self):
        self.numeric_columns = ['YoB','nevents','ndays_act','nplay_video','nchapters','nforum_posts']
        self.scalar = StandardScaler()

    def fit(self, X, y=None):
        self.numeric = X[self.numeric_columns]
        self.categorical = X.drop(self.numeric_columns, axis = 1)
        return self
    
    def transform(self, X):
        self.scalar.fit(self.numeric)
        num_scaled = pd.DataFrame(self.scalar.transform(self.numeric))
        num_scaled.rename(columns = dict(zip(num_scaled.columns,    self.numeric_columns)), inplace = True)
        return pd.concat([num_scaled, self.categorical], axis = 1)
    
class DropCols(BaseEstimator, TransformerMixin):
    '''
    Drop specified columns as features

    Parameters:
    ----------
    input {dataframe, list}: dataframe containing mixed featurevariable    types, list of names of numeric feature columns
    output: {dataframe}: dataframe with numeric features scaledand    categorical features unchanged

    '''
    def fit(self, X, y=None):
        self.cols_to_drop = ['course_id','userid_DI','start_time_DI','last_event_DI','YoB_imputed','nplay_video_imputed','nchapters_imputed','LoE_DI_imputed','gender_imputed', 'grade_imputed','grade', '1988.0_x', '1988.0_y']
        return self

    def transform(self, X):
        dropped = X.drop(self.cols_to_drop, axis=1)
        return dropped

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def print_roc_curve(y_test, probabilities, model_name, roc_auc, recall):
    '''
    Calculates and prints a ROC curve given a set of test classes and probabilities from a trained classifier
    
    Parameters:
    ----------
    y_test: 1d array
    probabilities: 1d array of predeicted probabilites from X_test data
    model_name: name of model, for printing plot label
    recall: model recall score
    '''
    tprs, fprs, thresh = roc_curve(y_test, probabilities)
    plt.figure(figsize=(12,10))
    plt.plot(fprs, tprs, 
         label="AUC: {}\nRecall {}".format(round(roc_auc, 3), round(recall, 3)),
         color='blue', 
         lw=3)
    plt.plot([0,1],[0,1], 'k:')
    plt.legend(loc = 4, prop={'size': 45})
    plt.xlabel("FPR", fontsize=20)
    plt.ylabel("TPR", fontsize=20)
    plt.title("ROC Curve: {}".format(model_name), fontsize=40)
    

######################################################################

if __name__ == '__main__':
    cd /Users/Jeremy/GoogleDrive/Data_Science/Projects/Education_Data/harvard_ed_x

    # read in the data
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    # y_train = y_train.values.ravel()

    # small test data sets
    # X_train = X_train[:100]
    # y_train = y_train[:100]

    d = DropCols()
    gb = GradientBoostingClassifier()#0.77

    p = Pipeline([
        ('drop', d),
        ('gb', gb)
    ])
    
    # first grid search
    # clf_params = {
    #         'gb__learning_rate': [0.1, 0.01, 0.001],
    #         'gb__n_estimators': [100, 500, 1000],
    #         'gb__subsample': [1, 0.5],
    #         'gb__min_samples_split': [2, 5, 10],
    #         'gb__min_samples_leaf': [1, 5, 10],
    #         'gb__max_depth': [2, 3, 4],
    #         'gb__max_features': ['sqrt', 'log2', 'auto'],
    # }

    # 2nd grid search
    clf_params = {
            'gb__learning_rate': [0.1],
            'gb__n_estimators': [1000],
            'gb__subsample': [0.5],
            'gb__min_samples_split': [5],
            'gb__min_samples_leaf': [1],
            'gb__max_depth': [6],
            'gb__max_features': ['auto'],
    }

    # gscv = RandomizedSearchCV(p, param_distributions=clf_params,
    #                     scoring='recall',
    #                     n_jobs=-1,
    #                     verbose=2,
    #                     return_train_score=True,
    #                     cv=5)

    gscv = GridSearchCV(p, param_grid=clf_params,
                        scoring='recall',
                        n_jobs=-1,
                        verbose=2,
                        cv=3)


    clf = gscv.fit(X_train, y_train)
    print(cross_validate(clf, X_train, y_train, return_train_score=True))

    # best params
    # {'gb__subsample': 0.5,
    # 'gb__n_estimators': 1000,
    # 'gb__min_samples_split': 5,
    # 'gb__min_samples_leaf': 1,
    # 'gb__max_features': 'auto',
    # 'gb__max_depth': 4,
    # 'gb__learning_rate': 0.1}

    # log_reg_model = lr_clf.best_estimator_

    # best model as determined by grid search
    # model = LogisticRegression(C=1, class_weight=None, dual=False,    fit_intercept=True, intercept_scaling=1, max_iter=200,            multi_class='warn', n_jobs=None, penalty='l2', random_state=None, solver='newton-cg', tol=0.0001, verbose=0, warm_start='False')

    # save model
    pickle.dump(clf, open('/Users/Jeremy/GoogleDrive/Data_Science/Projects/Education_Data/harvard_ed_x/models/gb_model.p', 'wb'))

    # # evaluation
    # predictions = log_reg_model.predict(X_test)
    # roc_auc = roc_auc_score(y_test, predictions)
    # probas = log_reg_model.predict_proba(X_test)[:, 1:]
    # tprs, fprs, thresh = roc_curve(y_test, probas)
    # recall = recall_score(y_test, predictions)
    # conf_mat = standard_confusion_matrix(y_test, predictions)
    # class_report = classification_report(y_test, predictions)

    # print_roc_curve(y_test, probas)
    # print('Best Model: {}'.format(log_reg_model))
    # print('Best Model parameters: {}'.format(lr_clf.best_params_))
    # print('Best Model Log Loss: {}'.format(lr_clf.best_score_))
    # print('Roc Auc: {}'.format(roc_auc))
    # print('Recall Score: {}'.format(recall))
    # print('Confusion Matrix: {}'.format(conf_mat))
    # print('Classification Report: {}'.format(class_report))

    # # Feature Importances
    # abs_coef = list(np.abs(log_reg_model.coef_.ravel()))
    # features = list(X_test.columns)
    # coef_dict = c.OrderedDict((zip(abs_coef, features)))
    # ordered_feature_importances = sorted(coef_dict.items(), reverse=True)
    # print('The top ten features affecting completion are: {}'.format( ordered_feature_importances[:10])
    