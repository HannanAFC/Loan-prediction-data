import pandas as pd

def loadCSV( ):
    df = pd.read_csv( "train.csv" )

    catagorical_fields = [ "Gender", "Dependents", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area", "Loan_Status" ]
    for catagorical_field in catagorical_fields:
        df[ catagorical_field ] = df[ catagorical_field ].fillna( df[ catagorical_field ].mode( )[ 0 ] )
        print( df[ catagorical_field ].mode() )

    numerical_fields = [ "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term" ]
    for numerical_field in numerical_fields:
        df[ numerical_field ] = df[ numerical_field ].fillna( df[ numerical_field ].median( ) )
        print( df[ numerical_field ].median() ) 

    df.to_csv( "train_cleansed.csv" )


if __name__ == "__main__":
    loadCSV( )