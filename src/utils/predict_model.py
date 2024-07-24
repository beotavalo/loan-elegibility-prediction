import joblib
import pandas as pd
import numpy as np

def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_csv(filename)
    return df

def feature_engineering(df_train:pd.DataFrame):
    #Filling missing values
    df_train['Gender'].fillna(df_train['Gender'].mode()[0], inplace=True)
    df_train['Married'].fillna(df_train['Married'].mode()[0], inplace=True)
    df_train['Dependents'].fillna(df_train['Dependents'].mode()[0], inplace=True)
    df_train['Self_Employed'].fillna(df_train['Self_Employed'].mode()[0], inplace=True)
    df_train['Credit_History'].fillna(df_train['Credit_History'].mode()[0], inplace=True)
    df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].mode()[0], inplace=True)
    df_train['LoanAmount'].fillna(df_train['LoanAmount'].median(), inplace=True)
    
    #Categorical variables encoding
    #df_train.Loan_Status = df_train.Loan_Status.replace({'Y': 1, 'N' : 0})
    df_train.Gender = df_train.Gender.replace({'Male': 1, 'Female' : 0})
    df_train.Married = df_train.Married.replace({'Yes': 1, 'No' : 0})
    df_train.Self_Employed = df_train.Self_Employed.replace({'Yes': 1, 'No' : 0})
    df_train.Education = df_train.Education.replace({'Graduate': 1, 'Not Graduate' : 0})
    df_train.Property_Area = df_train.Property_Area.replace({'Rural': 1, 'Semiurban' : 2,'Urban':3})
    df_train.Dependents = df_train.Dependents.replace({'0':0, '1':1, '2':2, '3+': 3})
    #Add new feature
    df_train['Total_Income']=df_train['ApplicantIncome'] + df_train['CoapplicantIncome']
    df_train['EMI']=df_train['LoanAmount']/df_train['Loan_Amount_Term']
    df_train['Balance Income'] = df_train['Total_Income']-(df_train['EMI']*1000)

    #Outliers treatment
    df_train['LoanAmount_log']=np.log(df_train['LoanAmount'])
    df_train['Total_Income_log'] = np.log(df_train['Total_Income'])

    return df_train

def feature_selection(df_train: pd.DataFrame):
    """
    First, I specify the model
    Then I use the selectFromModel object from sklearn, which
    will select the features which coefficients are non-zero"""
    #Select independent feature from dataset
    X = df_train[['Married', 'Education', 'Credit_History', 'Total_Income_log']]
    return X

def result_concatenate(df_result:pd.DataFrame, result):
    df_array = pd.DataFrame(result, columns=['Result'])
    # Convertir 1 a 'Sí' y 0 a 'No' usando apply y lambda
    df_array['Result'] = df_array['Result'].apply(lambda x: 'Approved' if x == 1 else 'Rejected')
    df_result = pd.concat([df_result, df_array], axis=1)
    return df_result

# Load the model
def load_model(model_path):
    model = joblib.load(model_path)
    return model 


# Make predictions
def main():
    data_path = '/workspaces/loan-elegibility-prediction/data/raw/loan-test.csv'
    model_path = '/workspaces/loan-elegibility-prediction/src/models/le_model.pickle'
    model= load_model(model_path)
    print('Step 1: Model loaded')
    
    #Data processing
    df_test = read_data(data_path)
    df_result = pd.DataFrame()
    df_result = df_test['Loan_ID']
    df_test = feature_engineering(df_test)
    df_test = feature_selection(df_test)

    print('Step 2: Data processed')
    result = model.predict(df_test)
    print('Step 3: Predicted')
    #Results
    df_result = result_concatenate(df_result,result)
    print(f'The results are:')
    print(df_result[0:20])

if __name__ == "__main__":
    main()
