import tkinter as tk
import api.tx_api as tx_api
import json

# Create a new window
window = tk.Tk()
# Set the aspect ratio of the window
window.geometry("300x600")
# Configure the window to use a grid layout
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Create a frame to hold the elements
frame = tk.Frame(window)
frame.pack(fill=tk.BOTH, expand=True)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Create an input field
input_field = tk.Entry(frame)
input_field.pack(side=tk.LEFT)

# Define a function to handle the button click event
def handle_click():
    # Get the text from the input field
    text = input_field.get()
    j = tx_api.tx_aip(text)
    data = json.loads(j)
    sentiment = data['Response']['Sentiment']
    # Convert the string to a slice
    # slice_text = slice(int(text))
    # Insert the slice into the text box
    # text_box.insert(tk.END, str(slice_text) + "\n"+"\n")
    # print(sentiment)
    # Insert the text into the text box
    text_box.insert(tk.END, text +"   "+ sentiment + "\n"+"\n")


# Create a buttonn
button = tk.Button(frame, text="Click Me", command=handle_click)
button.pack(side=tk.RIGHT)

# Create a text display box
text_box = tk.Text(window)
text_box.pack(fill=tk.BOTH, expand=True)
# Disable input field
# Add the frame to the window

frame.pack()
window.mainloop()
