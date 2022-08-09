from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing
from env import user, password, host

def get_db_url(database):
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

"""
USAGE: 
Use `from wrangle import wrangle_zillow` at the top of your notebook.
This 
"""
def get_zillow_data():
    """Seeks to read the cached zillow.csv first """
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_new_zillow_data()

def get_new_zillow_data():
    """Returns a dataframe of all 2017 properties that are Single Family Residential"""

    sql = """
    select 
    bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    where propertylandusedesc = "Single Family Residential"
    """
    return pd.read_sql(sql, get_db_url("zillow"))


def handle_nulls(df):    
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    return df


def optimize_types(df):
    # Convert some columns to integers
    # fips, yearbuilt, and bedrooms can be integers
    df["fips"] = df["fips"].astype(str)
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    return df


def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 1_500_000]

    df.drop(df.loc[df.calculatedfinishedsquarefeet >15000].index, inplace=True)

    df.drop(df.loc[df['bedroomcnt']==0].index, inplace=True)
    
    df.drop(df.loc[df['bathroomcnt']==0].index, inplace=True)

    return df


def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """
    df = get_zillow_data()

    df = handle_nulls(df)

    df = optimize_types(df)

    df = handle_outliers(df)

    #new column total bill divided by size
    df['cost_per_sqft'] = (df['taxvaluedollarcnt']/df['calculatedfinishedsquarefeet']).round(2)

    #avg sqft per number of rooms (bedroom+bathroom)
    df['sqft_room_ratio'] = (df['calculatedfinishedsquarefeet']/(df['bedroomcnt']+df['bathroomcnt'])).round(2)

    #dummy encode fips
    dummies = pd.get_dummies(df.fips)
    df = pd.concat([df,dummies],axis=1)

   # df.to_csv("zillow.csv", index=False)
   #rename columns for ease of use
    df=df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "sqft","taxvaluedollarcnt": "home_value"})

    return df

def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name for the stratify_by argument
    """

    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])

    return train, validate, test

def scale_zillow(train, validate, test):
    '''
    scale_zillow will 
    - fits a min-max scaler to the train split
    - transforms all three spits using that scaler. 
    returns: 3 dataframes with the same column names and scaled values. 
    '''
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    # Note that we only call .fit with the TRAINING data,
    scaler.fit(train)
    
    # but we use .transform to apply the scaling to all the data splits.    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # convert to arrays to pandas DFs
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    return train_scaled, validate_scaled, test_scaled






## TODO Encode categorical variables (and FIPS is a category so Fips to string to one-hot-encoding
## TODO Scale numeric columns
## TODO Add train/validate/test split in here
## TODO How to handle 0 bedroom, 0 bathroom homes? Drop them? How many? They're probably clerical nulls

# #new column total bill divided by size
# df['cost_per_sqft'] = (df['taxvaluedollarcnt']/df['calculatedfinishedsquarefeet']).round(2)

# #avg sqft per number of rooms (bedroom+bathroom)
# df['sqft_room_ratio'] = (df['calculatedfinishedsquarefeet']/(df['bedroomcnt']+df['bathroomcnt'])).round(2)


# (df.bedroomcnt == 0).value_counts()
# (df.bathroomcnt == 0).value_counts()
# (df['bedroomcnt']==0)&(df['bathroomcnt']==0)

# #at this point after digging into whether or not i can or should impute the 
# # values of the averages based on squarefootage or just drop them

# df.drop(df.loc[df['bedroomcnt']==0].index, inplace=True)
# df.drop(df.loc[df['bathroomcnt']==0].index, inplace=True)

# def select_kbest(X_train, y_train, k):
#     # parameters: f_regression stats test, give me 2 features
#     f_selector = SelectKBest(f_regression, k=k)
# # find the top 8 X's correlated with y
#     f_selector.fit(X_train, y_train)
# # boolean mask of whether the column was selected or not. 
#     feature_mask = f_selector.get_support()
# # get list of top K features. 
#     f_feature = X_train.iloc[:,feature_mask].columns.tolist()
#     return f_feature

# def rfe (X_train, y_train, k):
#     # initialize the ML algorithm
#     lm = LinearRegression()
# # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
#     rfe = RFE(lm, n_features_to_select=k)
# # fit the data using RFE
#     rfe.fit(X_train,y_train)  
# # get the mask of the columns selected
#     feature_mask = rfe.support_
# # get list of the column names. 
#     rfe_feature = X_train.iloc[:,feature_mask].columns.tolist()
#     return rfe_feature

#handle missing values with more than 13% nulls
def handle_missing_values(df, prop_required_columns =0.87, prop_required_row=0.87):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    #axis 1 : drop columns that have missing values
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns),0))
    #axis 0 : drop rows that have missing values
    df = df.dropna(axis=0, thresh=threshold)
    return df

#remove columns
def remove_columns(df, cols_to_remove):
    df = df.drop(columns = cols_to_remove)
    return df

# combining everything in a cleaning function:
def data_prep(df, cols_to_remove=[], prop_required_column=0.87, prop_required_row=0.87):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df    

def clean_and_prep(df):
##rename some columns for ease of use
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "sqft","taxvaluedollarcnt": "home_value"})
    
#remove homes with zero beds
    df = df[df.bedrooms != 0]
    
##filled null/Nonetype with 'None' or '0'
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('None')
    df.airconditioningdesc = df.airconditioningdesc.fillna('None')
    df.poolcnt = df.poolcnt.fillna('0')
    df.garagecarcnt = df.garagecarcnt.fillna('0')
    
#feature engineering
    df['home_value_structure_tax_difference'] = (df['home_value']-df['structuretaxvaluedollarcnt'])
    df['hvs_pct'] = (df['home_value_structure_tax_difference']/df['home_value'])
    df['taxrate'] = (df['taxamount']/df['home_value'])
    df['age'] = 2017 - df.yearbuilt
    df['cost_per_sqft'] = (df['home_value']/df['sqft']).round(2)
    df['sqft_room_ratio'] = (df['sqft']/(df['bedrooms']+df['bathrooms'])).round(2)
    mapping = {"None" : "0",
                     "Central" : "2",
                     "Yes" : "1",
                     "Wall Unit" : "1",
                     }
    df['airconditioning_encoded'] = df['airconditioningdesc'].map(mapping)

#impute nulls with tax rate mean times home value
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(df['home_value']*df['hvs_pct'].mean())
    
#impute 4 missing values for taxamount using mean taxrate of feature engineering
    df.taxamount = df.taxamount.fillna(df['home_value']*df['taxrate'].mean())
    
#impute nulls with regionidcity mode of 12447
    df['regionidcity'] = df.regionidcity.fillna(12447)
    
#impute missing values with below
    df.home_value_structure_tax_difference = df.home_value_structure_tax_difference.fillna(df['home_value']-df['structuretaxvaluedollarcnt'])
    
#impute missing values with below
    df.hvs_pct = df.hvs_pct.fillna(df['home_value_structure_tax_difference']/df['home_value'])
    
#impute nulls with mean year built
    df.yearbuilt = df.yearbuilt.fillna(df['yearbuilt'].mean())
    
#impute nulls with mode of regionidzip
    df.regionidzip = df.regionidzip.fillna(df.regionidzip.mode())
    
##replace nulls with median lot size
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.median())
    
##impute fullbathcnt with value from df.bathrooms
    df.fullbathcnt = df.fullbathcnt.fillna(df.bathrooms)                                                   
                                                       
##drop remaining handful of nulls
    df = df.dropna()                                                   
    return df

def optimize_types(df):
# Convert some columns to integers, optimize types
# # fips, yearbuilt, and bedrooms can be integers   
    df["fips"] = df["fips"].astype(str)
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    df["bedrooms"] = df["bedrooms"].astype(int)
    df["bathrooms"] = df["bathrooms"].astype(int)
    df["home_value"] = df["home_value"].astype(int)
    df["garagecarcnt"] = df["garagecarcnt"].astype(int)
    df["poolcnt"] = df["poolcnt"].astype(int)
#   df["heatingorsystemdesc"] = df["heatingorsystemdesc"].astype(int)
    df["airconditioning_encoded"] = df["airconditioning_encoded"].astype(int)
    df["sqft"] = df["sqft"].astype(int)
    df["home_value_structure_tax_difference"] = df["home_value_structure_tax_difference"].astype(int)
    df["taxamount"] = df["taxamount"].astype(int)
    df["landtaxvaluedollarcnt"] = df["landtaxvaluedollarcnt"].astype(int)
    df["structuretaxvaluedollarcnt"] = df["structuretaxvaluedollarcnt"].astype(int)
    df["regionidzip"] = df["regionidzip"].astype(int)
    df["lotsizesquarefeet"] = df["lotsizesquarefeet"].astype(int)
    return df        

# feature engineering luxuryscore will be counts for pool, garage, ac, number of bathrooms and bedrooms
df['luxury_score'] = (df['poolcnt']+df['garagecarcnt']+df['bedrooms']+df['bathrooms']+df['airconditioning_encoded'])

#feature engineering luxury score/age*sqft
df['luxury_sqft_per_age']= (df['luxury_score']*df['sqft']/df['age'])
df['luxury_sqft_per_age']=df['luxury_sqft_per_age'].astype(int)
#feature engineering
df['quarter'] = pd.PeriodIndex(df.transactiondate, freq='Q')
df['quarter']=df['quarter'].astype(str)

#dummy encode quarter
quarter_df = pd.get_dummies(df.quarter)
df = pd.concat([df, quarter_df],axis=1) 

def get_counties():
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies

def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.bathrooms <= 7]
    
    df = df[df.bedrooms <= 6]

    df = df[df.home_value < 3_000_000]

    df.drop(df.loc[df.sqft >10000].index, inplace=True)

    df.drop(df.loc[df['bedrooms']==0].index, inplace=True)
    
    df.drop(df.loc[df['bathrooms']==0].index, inplace=True)

    return df    

def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=13)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=13)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

#     partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return train, X_train, X_validate, X_test, y_train, y_validate, y_test    