# convert_data.py
import pandas as pd

columns = [
    'Status_of_existing_checking_account', 'Duration_in_month', 'Credit_history',
    'Purpose', 'Credit_amount', 'Savings_account_bonds', 'Present_employment_since',
    'Installment_rate_in_percentage_of_disposable_income', 'Personal_status_and_sex',
    'Other_debtors_or_guarantors', 'Present_residence_since', 'Property',
    'Age_in_years', 'Other_installment_plans', 'Housing', 'Number_of_existing_credits',
    'Job', 'Number_of_people_being_liable_to_provide_maintenance_for',
    'Telephone', 'foreign_worker', 'Credit_risk'
]

df = pd.read_csv("german.data", sep=" ", header=None, names=columns)
df.to_csv("german_credit_data.csv", index=False)
print("Converted to CSV successfully.")
