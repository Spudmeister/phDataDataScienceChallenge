import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURES_TO_EXCLUDE = ['i4', 'c10_no', 'c10_yes', 'c3_unknown', 'c4_new', 'b1_no', 'n4', 'b1_-1']

def load_data(input_path):
    #load raw data
    raw_df = pd.read_csv(input_path)
    
    #extract categorical columns
    categorical_df = raw_df.drop(columns=['successful_sell']).select_dtypes(include=['object'])
    
    #fill NaN values
    categorical_nan_filled_df = categorical_df.fillna('nan_filled')
    
    #one-hot encode
    encoded_categories_df = pd.get_dummies(categorical_nan_filled_df)
    
    
    #extract numerical columns
    numerical_df = raw_df.select_dtypes(include=['int','float'])
    
    #create standardscaler and transform numerical data
    sc = StandardScaler()
    sc.fit(numerical_df)
    scaled_values = sc.transform(numerical_df)
    
    scaled_numerical_df = pd.DataFrame(scaled_values, columns=numerical_df.columns)
    
    
    #combine numerical and categorical columns
    transformed_df = scaled_numerical_df.merge(encoded_categories_df, how='inner', left_index=True, right_index=True)\
        .drop(columns=FEATURES_TO_EXCLUDE)
    
    
    return raw_df.index.values, transformed_df.to_numpy()

def save_forecasts(ids, predictions, output_path):
    
    pd.DataFrame({'ids':ids, 'predictions':predictions}).to_csv(output_path)