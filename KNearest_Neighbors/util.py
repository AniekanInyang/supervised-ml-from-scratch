import pandas as pd

#Read in the data - loan prediction dataset
#change labels to numbers. N = 0 and Y = 1

#Drop Dependents/loan_amount_term, loan_id columns
#Remove rows with missing data
#Standardize/Normalize applicant/co-applicant income/loan amount column between 0 and 1
#Change property (Rural - 0, Semiurban - 0.5, Urban - 1)

def pre_process_data(data, limit=None):
    print ("Reading and processing data:")
    #read in data
    data = pd.read_csv(data)

    #convert these columns to 0 or 1
    data.loc[data['Loan_Status'] == "N", 'Loan_Status'] = 0
    data.loc[data['Loan_Status'] == "Y", 'Loan_Status'] = 1

    data.loc[data['Gender'] == "Male", 'Gender'] = 0
    data.loc[data['Gender'] == "Female", 'Gender'] = 1

    data.loc[data['Married'] == "No", 'Married'] = 0
    data.loc[data['Married'] == "Yes", 'Married'] = 1

    data.loc[data['Education'] == "Not Graduate", 'Education'] = 0
    data.loc[data['Education'] == "Graduate", 'Education'] = 1

    data.loc[data['Self_Employed'] == "No", 'Self_Employed'] = 0
    data.loc[data['Self_Employed'] == "Yes", 'Self_Employed'] = 1

    #drop irrelevant columns for this project
    data = data.drop(columns = ['Dependents', 'Loan_Amount_Term', 'Loan_ID'])

    #converting property area data to between 0 and 1
    data.loc[data['Property_Area'] == "Rural", 'Property_Area'] = 0
    data.loc[data['Property_Area'] == "Semiurban", 'Property_Area'] = 0.5
    data.loc[data['Property_Area'] == "Urban", 'Property_Area'] = 1

    #removing rows with missing values
    data = data.dropna()
    #return data.isnull().values.any()

    #Tip: Which Method To Use (standardizing or normalizing?)
    #It is hard to know whether rescaling your data will improve the performance of your algorithms before you apply them. 
    #If often can, but not always.
    #A good tip is to create rescaled copies of your dataset and 
    #race them against each other using your test harness and a handful of algorithms you want to spot check. 
    #This can quickly highlight the benefits (or lack there of) of rescaling your data with given models, 
    #and which rescaling method may be worthy of further investigation.

    #To learn, normalize these columns (MinMax scaler) and standardize (mean and sd), then compare performance

    data['ApplicantIncome']=(data['ApplicantIncome']-data['ApplicantIncome'].min())/(data['ApplicantIncome'].max()-data['ApplicantIncome'].min())
    data['CoapplicantIncome']=(data['CoapplicantIncome']-data['CoapplicantIncome'].min())/(data['CoapplicantIncome'].max()-data['CoapplicantIncome'].min())
    data['LoanAmount']=(data['LoanAmount']-data['LoanAmount'].min())/(data['LoanAmount'].max()-data['LoanAmount'].min())

    return data

#test function
check_data = pre_process_data('loan_data.csv')
print (check_data)

