import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.width', 1000)
from sklearn.model_selection import train_test_split

def check_reg(df):
    '''
    A check to make sure that all student in the data are actually registered in a course.

    Parameters:
    ----------
    input {df}: Pandas dataframe
    output {}: None
    '''
    a = df.shape[0]
    b = df[df["registered"]==1].shape[0]
    assert a == b, "Some unregistered students, check data!"
    print("\nNo records of unregistered students")

def unique_rows(df):
    '''
    Drop incomplete rows for users with duplicate rows who have at least one complete row.

    Parameters:
    ----------
    input {df}: Pandas dataframe
    output: None
    '''
    #find users with more than one row
    vc = df['userid_DI'].value_counts()
    # series of ids of students more than one course
    more_than_one = vc[vc > 1] 
    # array of ids of students with more than one course
    ids = more_than_one.index.values
    # drop any rows with any nans of people with more than one
    # only of there is at least one
    for num in ids:
        temp = df[df['userid_DI'] == num]
        temp = temp[['ndays_act', 'nplay_video', 'nchapters', 'nforum_posts']]
        mask = temp.isnull().any(axis=1)
        if temp[~mask].shape[0] > 0: # if at least one full row
            indices = temp[mask].index.values # where any nulls
            df.drop(indices, axis=0, inplace=True) 
    
    return None

def check_one_row_per(df):
    '''
    A check to make sure that all student in the data are actually registered in a course

    Parameters:
    ----------
    input {df}: Pandas dataframe
    output: None
    '''
    a = df.shape[0]
    b = df["userid_DI"].nunique()
    assert a == b, "Some students appear more that onec, check data!"
    print("\nNo duplicated students records.")

def impute_median(df, columns):
    '''
    Impute null values with mean for specified columns; add boolean column indicating value was imputed.
    
    Parameters:
    ----------
    input:
    df: Pandas dataframe
    columns: list of columns to impute

    output: None
    '''

    for col in columns:
        imputed = df[col].isnull()
        val = df[col].median()
        df.fillna(value=val, axis=1, inplace=True)
        name = str(col + "_imputed")
        df[name] = imputed
    return None

def nan_to_unknown(df, columns):
    '''
    Impute null values to string 'Unknown', creating a new category definition for a categorical variable.
    
    Parameters:
    ----------
    input:
    df: Pandas dataframe
    columns: list of columns to impute

    output: None
    '''

    for col in columns:
        imputed = df[col].isnull()
        val = "Unknown"
        df.fillna(value=val, axis=1, inplace=True)
        name = str(col + "_imputed")
        df[name] = imputed
    return None

def nan_to_zero(df, columns):
    '''
    Impute null values to zero in specified columns.

    Parameters:
    ----------
    input:
    df: Pandas dataframe
    columns: list of columns to impute

    output: None
    '''
    # change to 0 if no grade
    for col in columns:
        imputed = df[col].isnull()
        df.fillna(value=0, axis=1, inplace=True)
        name = str(col + "_imputed")
        df[name] = imputed
    return None

def one_hot(input_df, columns):
    """
    Returns a dataframe with categorical columns transformed to dummy variables.

    Parameters:
    ----------
    input:
    df: Pandas dataframe
    columns: list of columns to impute

    output: Pandas dataframe
    """
    df = input_df.copy()

    for col in columns:
        dummies = pd.get_dummies(df[col].str.lower())
        dummies.drop(dummies.columns[-1], axis=1, inplace=True)
        df = df.drop(col, axis=1).merge(dummies, left_index=True, right_index=True)
    
    return df

##################################################################
if __name__ == "__main__":
    cd /Users/Jeremy/GoogleDrive/Data_Science/Projects/Education_Data/harvard_ed_x

    df = pd.read_csv("data/HMXPC13_DI_v2_5-14-14.csv")
    # df = pd.read_csv("data/cleaned_harvard_data.csv")

    # check that every record is "registered"
    check_reg(df)

    # All values for registered are "1"; can drop column.
    df = df.drop("registered", axis=1)

    # Only keep records with at least one event
    df = df[df["nevents"] > 0]

    # drop 'roles', and 'incomplete flag'
    # documentation indicates that they are artifacts and not relevant data
    df = df.drop("roles", axis=1)
    df = df.drop("incomplete_flag", axis=1)

    # drop incompete duplicates
    unique_rows(df)

    # drop duplicates by userid
    df.drop_duplicates(subset="userid_DI", inplace=True)

    # impute median for specified columns
    impute_median(df, ['YoB', 'nplay_video', 'nchapters'])

    # impute to unkown for specific categorial columns
    nan_to_unknown(df, ['LoE_DI', 'gender'])

    # impute grade to zero
    nan_to_zero(df, ["grade"])
    
    df_dums = one_hot(df, ['final_cc_cname_DI', 'LoE_DI', 'gender'])

    #split data
    X = df_dums.drop('certified', axis=1)
    y = df_dums[['certified']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # write out cleaned data files
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)  
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)


