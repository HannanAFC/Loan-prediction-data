import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_csv( ):
    df = pd.read_csv( "train.csv" )

    catagorical_fields = [ "Gender", "Dependents", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area", "Loan_Status" ]
    numerical_fields = [ "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term" ]

    df = replace_missing_data( df, catagorical_fields, numerical_fields )
    df = remove_outliers( df, numerical_fields )
    df = hot_encode( df, catagorical_fields )
    df = normalise( df, numerical_fields )

    df.to_csv( "train_cleansed.csv" )

def replace_missing_data( df: pd.DataFrame, catagorical_fields: list, numerical_fields: list ):
    #replace missing values for catagorical fields
    for catagorical_field in catagorical_fields:
        df[ catagorical_field ] = df[ catagorical_field ].fillna( df[ catagorical_field ].mode( )[ 0 ] )
        #print( df[ catagorical_field ].mode() )

    #replace missing values for numerical fields
    for numerical_field in numerical_fields:
        df[ numerical_field ] = df[ numerical_field ].fillna( df[ numerical_field ].median( ) )
        #print( df[ numerical_field ].median() )

    return df

def remove_outliers( df: pd.DataFrame, numerical_fields: list ):
    #find and remove outliers for numerical fields
    for numerical_field in numerical_fields:
        Q1 = df[ numerical_field ].quantile( 0.25 )
        Q3 = df[ numerical_field ].quantile( 0.75 )
        IQR = Q3 - Q1
        Lower_Bound = Q1 - ( 1.5 * IQR )
        Upper_Bound = Q3 + ( 1.5 * IQR )
        #print( df[ ( ( df[ numerical_field ] < Lower_Bound ) | ( df[ numerical_field ] > Upper_Bound ) ) ] )
        df = df[ ~( ( df[ numerical_field ] < Lower_Bound ) | ( df[ numerical_field ] > Upper_Bound ) ) ]
    
    return df

def hot_encode( df: pd.DataFrame, catagorical_fields: list ):
    #convert categorical values into integers
    l1 = LabelEncoder()
    for catagorical_field in catagorical_fields:
        l1.fit( df[ catagorical_field ] )
        df[ catagorical_field ] = l1.transform( df[ catagorical_field ] )
        #print( df[ catagorical_field ] )
    
    return df

def normalise( df: pd.DataFrame, numerical_fields: list ):
    #normalise values in numerical fields
    ss = StandardScaler()
    for numerical_field in numerical_fields:
        #note double brackets, selects as a dataframe (required by scikit)
        df[ numerical_field ] = ss.fit_transform( df[ [numerical_field ] ] )

    return df

if __name__ == "__main__":
    load_csv( )