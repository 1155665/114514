import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import bd_api

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
print(cleaned_data)





''''
# Create a new window
window = tk.Tk()

# Add GUI elements and define their behavior

# Define the behavior 
def my_function():
    # Code to be executed when the button is clicked
    print("Button clicked!")
# Create a button
button = tk.Button(window, text="Click Me", command=my_function)
# Add the button to the window
button.pack()
# Run the GUI event loop
window.mainloop()
'''