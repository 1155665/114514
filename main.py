import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import api.bd_api as bd_api
from api.tx_api import *

import pretty_errors
pretty_errors.activate()


# Import the CSV file
data = pd.read_csv('/Users/surui/Downloads/data.csv')
# Import the second column
second_column = data.iloc[:, 1]

# Clean the data by removing empty values
cleaned_data = second_column.dropna()
# Remove duplicate values from the cleaned data
cleaned_data = cleaned_data.drop_duplicates()
# Remove the specified text from the cleaned data
cleaned_data = cleaned_data.str.replace('\r\n\t\t\t\t\t', '')
cleaned_data = cleaned_data.str.replace('病情描述（发病时间、主要症状、症状变化等）：', '')
cleaned_data = cleaned_data.str.replace('体格检查：', '')
cleaned_data = cleaned_data.str.replace('影像检查：', '')

# Output the cleaned data without duplicate values
# print(cleaned_data)

# Write the cleaned data to the 29th column
data.insert(32, "CLEANED", cleaned_data)
#data.iloc[:, 32] = cleaned_data
# Save the updated file
data.to_csv('/Users/surui/Downloads/updated_data.csv', index=False)

