import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

def general_clean( df ):
    #Random_state is used to set the seed for the random generator so that we can
    #ensure that the results that we get can be reproduced.
    df.sample( 5, random_state=42 )
    #Since we want to create our own clusters, let’s remove this column along with ID
    df = df.drop( [ "Loan_ID" ], axis="columns" )
    df.head( )
    df.info( )
    #For the sake of simplicity, we remove all rows with any missing values with
    df = df.dropna( )
    df.head( )
    df.info( )
    #reset it with df.reset_index(), then remove the freshly created index column
    df = df.reset_index( )
    df = df.drop( "index", axis="columns" )
    df.head( )

    #convert categorical values to numerical data
    df = pd.get_dummies( df )
    df.head( )
    df = pd.get_dummies( df, drop_first=True )
    df.head( )
    return df

def kmeans_clustering( file_path ):
    df = pd.read_csv( file_path )
    df_kmeans = general_clean( df )
    #model creastion
    kmeans_model = KMeans( n_clusters=3 )
    #data clustering
    clusters = kmeans_model.fit_predict( df_kmeans )

    #insert the cluster label
    df_kmeans.insert( df_kmeans.columns.get_loc( "ApplicantIncome" ), "Cluster", clusters )
    df_kmeans.head( 3 )
    #cluster labels
    df_kmeans.Cluster.unique( )

    numeric_cols = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]
    print( df_kmeans.groupby( "Cluster" )[ numeric_cols ].mean( ) )

    plt.figure( )
    plt.scatter(
        df_kmeans[ "ApplicantIncome" ],
        df_kmeans[ "LoanAmount" ],
        c=df_kmeans[ "Cluster" ]
    )
    plt.xlabel( "Applicant Income" )
    plt.ylabel( "Loan Amount" )
    plt.title( "K-Means Clusters: Income vs Loan Amount" )
    plt.show( )
    

def gmm_clustering( file_path ):
    df = pd.read_csv( file_path )
    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]

    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ]

    df = general_clean( df )

    gmm = GaussianMixture( n_components=3, random_state=42 )
    df[ "GMM_Cluster" ] = gmm.fit_predict( df )

    print( df.groupby( "GMM_Cluster" )[ numeric_cols ].mean( ) )

    plt.figure( )
    plt.scatter(
        df[ "ApplicantIncome" ],
        df[ "LoanAmount" ],
        c=df[ "GMM_Cluster" ]
    )
    plt.xlabel( "Applicant Income" )
    plt.ylabel( "Loan Amount" )
    plt.title( "GMM Clusters: Income vs Loan Amount" )
    plt.show( )


if __name__ == "__main__":
    load_csv( )
    gmm_clustering( "train.csv" )
    kmeans_clustering( "train.csv" )