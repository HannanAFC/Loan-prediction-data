import pandas as pd
from IPython.core.display_functions import display
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from six import StringIO  
from IPython.display import Image  
import pydotplus
import graphviz

def load_csv( file_path ):
    df = pd.read_csv( file_path )
    return df

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

def cleanse_data( df: pd.DataFrame ):
    catagorical_fields = [ "Gender", "Dependents", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area", "Loan_Status" ]
    numerical_fields = [ "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term" ]

    df = replace_missing_data( df, catagorical_fields, numerical_fields )
    df = remove_outliers( df, numerical_fields )
    df = hot_encode( df, catagorical_fields )
    df = normalise( df, numerical_fields )

    df.to_csv( "train_cleansed.csv", index=False )

def decision_tree( df: pd.DataFrame ):
    df = df.drop( columns="Loan_ID", axis=1 )
    feature_columns = list( df.drop( columns="Loan_Status", axis=1 ).columns.values )
    #feature columns
    X = df[ feature_columns ]
    #target column
    y = df.Loan_Status

    #split into training and test data sets, 30% testing size
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1 )
    clf = DecisionTreeClassifier( )
    clf = clf.fit( X_train,y_train )
    y_pred = clf.predict( X_test )
    print( "Accuracy:", metrics.accuracy_score( y_test, y_pred ) )

    dot_data = StringIO( )

    export_graphviz(
        clf, out_file=dot_data,
        filled=True, rounded=True,
        special_characters=True,feature_names = feature_columns,class_names=[ "0","1" ]
    )

    graph = pydotplus.graph_from_dot_data( dot_data.getvalue( ) )  
    graph.write_png( "loan_prediction_dt.png" )
    Image( graph.create_png( ) )
    

def random_forest( df: pd.DataFrame ):
    df = df.drop( columns=[ "Loan_ID" ] )
    
    #X for features, y for target
    X = df.drop( "Loan_Status", axis=1 )
    y = df[ "Loan_Status" ]

    #split into training and test data sets, 20% testing size
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

    rf = RandomForestClassifier( )
    rf.fit( X_train, y_train )

    y_pred = rf.predict( X_test )

    accuracy = accuracy_score( y_test, y_pred )
    print( "Accuracy:", accuracy )

    #export first three decision trees
    for i in range( 3 ):
        tree = rf.estimators_[ i ]
        dot_data = export_graphviz(
            tree,
            feature_names=X_train.columns,
            filled=True,
            max_depth=3,
            impurity=False,
            proportion=True
        )
        graph = graphviz.Source(dot_data)
        graph.render( "Loan_Prediction_Random_Forest", format="png", cleanup=True )

if __name__ == "__main__":
    df = load_csv( "train.csv"  )
    cleanse_data( df )
    df = load_csv( "train_cleansed.csv" )
    decision_tree( df )
    random_forest( df )