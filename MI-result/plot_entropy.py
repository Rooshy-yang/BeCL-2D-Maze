import matplotlib.pyplot as plt 
import numpy as np


algo_dict = {
    "cic_mi": "CIC",
    "contrastive_mi_0.5": "BeCL-tem0.5",
    "contrastive_mi_0.01": "BeCL-tem0.01",
    "reverse_mi": "Reverse MI",
    "forward_mi": "Forward MI",
    }

entropy_dict = {
    "contrastive_mi_0.5": 1.7443,
    "contrastive_mi_0.01": 2.0693,
    "cic_mi": 2.2904,
    "reverse_mi": 1.3789,
    "forward_mi": 1.6947,
}

algo_list = list(entropy_dict.keys())
label_list = []
for algo in algo_list:
    label_list.append(algo_dict[algo])

entropy_list = list(entropy_dict.values())

plt.bar(range(len(algo_list)), entropy_list, width=0.5, tick_label=label_list)
plt.title("Entropy Estimation")
plt.savefig(f"Entropy-compare.jpg", dpi=150)
print("done")




