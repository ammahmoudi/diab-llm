import pickle
import matplotlib.pyplot as plt

# Load the lists from pickle file
with open("./logs/logs_2025-02-02_11-17-53/loss.pkl", "rb") as f:
    list1, list2 = pickle.load(f)

# Ensure both lists have the same length
if len(list1) != len(list2):
    raise ValueError("Lists must have the same length for plotting.")

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(list1, label="train loss", marker='o', linestyle='-')
plt.plot(list2, label="validation loss", marker='s', linestyle='--')

# Customize the plot
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Plot of Two Lists from Pickle File")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("plot.png", dpi=300)  # Save as PNG with high resolution
plt.close()  # Close the figure to free memory

print("Plot saved as 'plot.png'")
