#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder 
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, roc_auc_score, cohen_kappa_score, accuracy_score, adjusted_mutual_info_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, chi2, f_classif,f_regression, SelectFromModel, RFE, mutual_info_classif, SelectPercentile, mutual_info_regression
from sklearn.linear_model import Lasso
from boruta import BorutaPy

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor, VotingClassifier
from sklearn.tree import  DecisionTreeClassifier, DecisionTreeRegressor


# ------------------------------------------------------
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem)) 
    return df

# ------------------------------------------------------
# Drop columns with missing values > 85% 
def droping_columns(df):
    Missing = df.isna().mean()*100
    colums_to_drop = df.columns[Missing>85]
    df.drop(columns = colums_to_drop, inplace=True)
    return df

get_droping_columns = FunctionTransformer(droping_columns)

# ------------------------------------------------------
# Replace all missing values in categorical columns with 'Unknown', allowing to retain as many columns as possible for analysis.
def cleaning(df):
    threshold = 101   
    for i in df.select_dtypes(include=['category']).columns:
        df[i] = df[i].astype('category')
        df[i] = df[i].cat.add_categories(['missing', 'noise'])        
        df[i] = df[i].fillna(df[i].mode()[0])  
    
        count = df[i].value_counts(dropna=False)
        less_freq = count[count < threshold].index
        
        df[i] = df[i].apply(lambda x: 'noise' if x in less_freq else x)   
    return df

get_cleaning = FunctionTransformer(cleaning)

# ------------------------------------------------------
# Handle the Outliers
# The class of the function that will perform the winsorization on the data based on the parameters learnt during the fit process
class Custom_winsorize(): 

    # The constructor of the class that will initialize the properties of the object 
    def __init__(self): 
        
        # Numeric features in the data 
        self.numeric_cols = []
        
        # Store the lower and the upper whisker values for each feature in the data 
        # A tuple for each column would be stored in a dictionary
        self.whiskers = dict()
        
    # This function would help in order to identify the lower and the upper bound of the whiskers 
    def outlier_bound(self, X, column):
    
        # Getting the 25th percentile of the column 
        q1= np.percentile(X[column], 25)

        # Getting the 75th percentile of the column 
        q3= np.percentile(X[column], 75)

        # Computing the IQR for the data 
        iqr=q3-q1

        # Getting the upper whisker of the data
        upper_whisker = q3 + 1.5*(iqr)

        # Getting the lowe whisker of the data
        lower_whisker = q1 - 1.5*(iqr)

    
        # Return the upper and the lowe whisker of the data 
        return (round(lower_whisker, 2), round(upper_whisker, 2))

    # This function would help in order to identify the whiskers value for each column in the data 
    def fit(self, X, y = None ): 
        
        # Convert the data to a dataframe to make sure that we are working on a dataframe
        X = pd.DataFrame(X)
        
        # Identify the numeric features in the data
        self.numeric_cols = [col for col in X.columns if X[col].dtype != 'object']
        
        # Get the whiskers bound for each feature in the data 
        for column in self.numeric_cols:
        
            # Get the upper and the lower value of the whiskers 
            self.whiskers[column] = self.outlier_bound(X, column)
        

        # Return the object itself 
        return self
    
    # Transform the data based on the found whiskers values 
    def transform(self, X, y = None):
    
        # Take a copy of the initial data 
        Temp_X = pd.DataFrame(X).copy()
        
        # Loop over the numeric features that we have in the data 
        for column in self.numeric_cols:
            
            # Take the lower and the upper bound of the whiskers for this column 
            lower, upper = self.whiskers[column]
            
            # Get the percentage of the samples that should be winsorized in the lower bound
            limit_lower_percent = Temp_X[(Temp_X[column] < lower)].shape[0] / Temp_X.shape[0]
            
            # Get the percentage of the sampels that should be winsorized in the upper bound
            limit_upper_percent = Temp_X[(Temp_X[column] > upper)].shape[0] / Temp_X.shape[0]
            
            # Applying the winsorization given the parameters for this column
            Temp_X[column] = winsorize(Temp_X[column], limits=[limit_lower_percent, limit_upper_percent])
            
        # Return the winsorized version of the data
        return Temp_X

get_customwinsorize = Custom_winsorize()

# ------------------------------------------------------
# Handle the Skeweness
def apply_boxcox(df):
    columns=df.select_dtypes(include=['number']).columns.tolist()
    df_transformed = df.copy()
    
    for col in columns:
        # Ensure the data is strictly positive
        if (df[col] > 0).any():
            # Shift the data if there are zero or negative values
            shift = abs(df[col].min()) + 1
            df_transformed[col] = df[col] + shift
        else:
            shift = 0        
        df_transformed[col], best_lambda = boxcox(df_transformed[col])
        
    return df_transformed

get_apply_boxcox = FunctionTransformer(apply_boxcox)

# ------------------------------------------------------
# Handle the Outliers
def handle_outliers(df):

    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    #clip: everything smaller than lower_bound = lower_bound / everything grater than upper_bound = upper_bound
    return df

get_handle_outliers = FunctionTransformer(handle_outliers)

# ------------------------------------------------------
# Aggregate numerical and categorical
num_cols = make_column_selector(dtype_include='number')
cat_cols = make_column_selector(dtype_exclude='number')

# Numeric Columns - imputing missing values
imp_median = SimpleImputer(strategy='median', add_indicator=True)
imp_knn = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

# Categorical Columns - imputing missing values
imp_constant = SimpleImputer(strategy='constant', fill_value='missing')


# ohe = OneHotEncoder(handle_unknown='ignore')
# The problem is that this representation includes redundancy
# For example, if we know that [1, 0, 0] represents “blue” and [0, 1, 0] represents “green” 
# we don’t need another binary variable to represent “red“, 
# instead we could use 0 values for both “blue” and “green” alone, e.g. [0, 0].
ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
ori = OrdinalEncoder()
minmaxscaler = MinMaxScaler()
scaler = StandardScaler()
# power = PowerTransformer(method='box-cox', standardize=True) #only works with strictly positive values
power = PowerTransformer(method='yeo-johnson', standardize=True) #default-works with positive and negative values





preprocessor_1 = make_column_transformer(
                                        (make_pipeline(imp_median,get_customwinsorize, minmaxscaler), num_cols),
                                        (make_pipeline(imp_constant, ohe), cat_cols)  
                                      )
preprocessor_2 = make_column_transformer(
                                        (make_pipeline(imp_median,scaler), num_cols),
                                        (make_pipeline(imp_constant, ohe), cat_cols)
                                      )
preprocessor_3 = make_column_transformer(
                                        (make_pipeline(get_droping_columns, imp_median,scaler), num_cols),
                                        (make_pipeline(get_droping_columns,get_cleaning, imp_constant, ohe), cat_cols)
                                      )
preprocessor_4 = make_column_transformer(
                                        (make_pipeline(get_droping_columns, imp_median,scaler), num_cols),
                                        (make_pipeline(get_droping_columns, imp_constant, ohe), cat_cols)
                                      )

preprocessor_5 = make_column_transformer(
                                        (make_pipeline(get_droping_columns, imp_median, get_customwinsorize, minmaxscaler, power), num_cols),
                                        (make_pipeline(get_droping_columns, imp_constant, ohe), cat_cols)
                                      )

# ------------------------------------------------------
# Feature selection
# Lasso Regularization
selection_0 = SelectFromModel(Lasso(alpha=0.01))
# SelectPercentile default f_classify
selection_1 = SelectPercentile(chi2, percentile=90)
selection_2 = SelectKBest(chi2, k=20)
selection_3 = SelectKBest(f_classif)
# ANOVA F-test in the f_classif() f_regression() function.
selection_4 = SelectKBest(f_regression)
selection_5 = SelectFromModel(Lasso(alpha=0.001))
#  RFE
selection_6 = RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=10)
selection_7 = RFE(DecisionTreeRegressor(random_state=42), n_features_to_select=1)
selection_8 = RFE(RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42), n_features_to_select=10)
#  RFECV
selection_9 = RFECV(estimator=RandomForestRegressor(), min_features_to_select=30)
selection_10 = RFECV(estimator=RandomForestRegressor(random_state=42), step=1, cv=10, scoring='r2')
# min_features_to_select — The minimum number of features to be selected.
selection_11 = SelectPercentile(mutual_info_regression, percentile=90)
selection_12 = SelectKBest(mutual_info_classif, k=50)
# Boruta
selection_13 = BorutaPy(RandomForestRegressor(max_depth = 5), n_estimators='auto', verbose=2, random_state=42)
selection_14 = BorutaPy(RandomForestRegressor(max_depth=5),n_estimators='auto', verbose=2, random_state=42, max_iter=50, perc=90)


# ------------------------------------------------------
def object_to_boolen(df):
    object_to_boolen = [i for i in df.columns if df[i].nunique() < 3]        
    for j in object_to_boolen:
         df[j] = df[j].astype('bool')
    return df
# ------------------------------------------------------
def feature_engineering(df):
    season_cols = [col for col in df.columns if 'Season' in col]
    df = df.drop(season_cols, axis=1) 
    df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
    df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
    df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
    df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
    df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
    df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
    df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
    df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
    df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
    df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
    df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
    
    return df
# ------------------------------------------------------
def convert_types (df):
    object_to_categorical = df.select_dtypes(include=['object'])
    numerical_int = df.select_dtypes(include=['int64'])
    numerical_float = df.select_dtypes(include=['float64'])
    
    for i in object_to_categorical:
         df[i] = df[i].astype('category')
    for i in numerical_int:
         df[i] = df[i].astype('int32')  
    for i in numerical_float:
         df[i] = df[i].astype('float32') 
    return df

# ------------------------------------------------------
def convert_types_2 (df): 
    for i in X.select_dtypes(include=['float64']):
         df[i] = df[i].astype('float32') 
    for i in X.select_dtypes(include=['int64']):
         df[i] = df[i].astype('int32')         
    return df


# ------------------------------------------------------
def import_data(file):
    """Load a CSV file into a DataFrame and optimize memory usage."""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

# ------------------------------------------------------
def preprocess_data(file, drop_columns=True):
    """Complete preprocessing pipeline."""
    df = import_data(file)
    df['sii'] = df['sii'].fillna(df['sii'].mode()[0])
#     df = object_to_boolen(df)
    df = feature_engineering(df)
#     df = dataset_stabilizer(df)
#     if drop_columns:
#         df = droping_columns(df)
#     df = cleaning(df)
    df = convert_types(df)
#     df = apply_boxcox(df)
    df = handle_outliers(df)
#     df = convert_types_2(df)
    return df

