import numpy as np
import matplotlib.pyplot as plt
from data_processing.data_sets import Dataset_T1DM

data = np.array([[127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190],
                 [127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190],
                 [127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190],
                 [127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190],
                 [127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190],
                 [127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190],
                 [127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190],
                 [127], [123], [118], [115], [119], [121], [125], [130], [135], [140], 
                 [145], [150], [155], [160], [165], [170], [175], [180], [185], [190]])

# Apply missingness (for example, 30% missing data with random missingness)
miss_rate = 0.1
missing_type = 'random'  # Choose 'random', 'synthetic', or 'periodic'
data_imputed = Dataset_T1DM.apply_missingness(data, miss_rate, missing_type)
print(type(data))
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Data', color='blue', marker='o')
plt.plot(data_imputed, label='Data with Missingness & Imputation (Zero)', color='red', linestyle='--', marker='x')
plt.title(f"Data with {int(miss_rate * 100)}% Missingness ({missing_type.capitalize()}) and Zero Imputation")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("rand.jpg")