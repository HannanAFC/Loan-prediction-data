import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def loadCSV( ):
    df = pd.read_csv( "train.csv" )
    l1 = LabelEncoder()
    ss = StandardScaler()

    catagorical_fields = [ "Gender", "Dependents", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area", "Loan_Status" ]
    numerical_fields = [ "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term" ]

    #replace missing values for catagorical fields
    for catagorical_field in catagorical_fields:
        df[ catagorical_field ] = df[ catagorical_field ].fillna( df[ catagorical_field ].mode( )[ 0 ] )
        #print( df[ catagorical_field ].mode() )

    #replace missing values for numerical fields
    for numerical_field in numerical_fields:
        df[ numerical_field ] = df[ numerical_field ].fillna( df[ numerical_field ].median( ) )
        #print( df[ numerical_field ].median() )

    #find and remove outliers for numerical fields
    for numerical_field in numerical_fields:
        Q1 = df[ numerical_field ].quantile( 0.25 )
        Q3 = df[ numerical_field ].quantile( 0.75 )
        IQR = Q3 - Q1
        Lower_Bound = Q1 - ( 1.5 * IQR )
        Upper_Bound = Q3 + ( 1.5 * IQR )
        #print( df[ ( ( df[ numerical_field ] < Lower_Bound ) | ( df[ numerical_field ] > Upper_Bound ) ) ] )
        df = df[ ~( ( df[ numerical_field ] < Lower_Bound ) | ( df[ numerical_field ] > Upper_Bound ) ) ]

    #convert categorical values into integers
    for catagorical_field in catagorical_fields:
        l1.fit( df[ catagorical_field ] )
        df[ catagorical_field ] = l1.transform( df[ catagorical_field ] )
        #print( df[ catagorical_field ] )

    #normalise values in numerical fields
    for numerical_field in numerical_fields:
        #note double brackets, selects as a dataframe (required by scikit)
        df[ numerical_field ] = ss.fit_transform( df[ [numerical_field ] ] )

    df.to_csv( "train_cleansed.csv" )


if __name__ == "__main__":
    loadCSV( )