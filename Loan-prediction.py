import pandas as pd

def loadCSV( ):
    df = pd.read_csv( "train.csv" )

    #replace missing values for catagorical fields
    catagorical_fields = [ "Gender", "Dependents", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area", "Loan_Status" ]
    for catagorical_field in catagorical_fields:
        df[ catagorical_field ] = df[ catagorical_field ].fillna( df[ catagorical_field ].mode( )[ 0 ] )
        #print( df[ catagorical_field ].mode() )

    #replace missing values for numerical fields
    numerical_fields = [ "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term" ]
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

    df.to_csv( "train_cleansed.csv" )


if __name__ == "__main__":
    loadCSV( )