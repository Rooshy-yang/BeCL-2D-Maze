import matplotlib.pyplot as plt 
import numpy as np


def ma(a, window_size=100):
    # Average the entire result with a sliding window
    return [np.mean(a[i: i+window_size]) for i in range(0, len(a)-window_size)]


algo_dict = {
    "cic_mi": "CIC",
    "contrastive_mi_0.5": "BeCL-tem0.5",
    "contrastive_mi_0.01": "BeCL-tem0.01",
    "reverse_mi": "Reverse MI",
    "forward_mi": "Forward MI",
    }
    
algo_list = list(algo_dict.keys())

for algo in algo_list:
    result_indep = np.load(f"MI-{algo}.npy")
    n = result_indep.shape[0]
    result_indep = result_indep[:n//2]
    # plot
    result_indep_ma = ma(result_indep)
    print("Mutual Information:", result_indep_ma[-1])
    plt.plot(range(len(result_indep_ma)), result_indep_ma, label=algo_dict[algo])

plt.legend()
plt.grid('on')
plt.title("MINE Estimation")
plt.xlabel('MINE iteration')
plt.ylabel('mutual information')
plt.savefig(f"MI-compare.jpg", dpi=150)







