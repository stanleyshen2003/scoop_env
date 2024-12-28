import numpy as np
import matplotlib.pyplot as plt

def calculate_threshold(probs, epsilon=0.1):
    probs = np.array(probs)
    probs = 1 - probs  
    q_level = np.ceil((len(probs) + 1 ) * (1 - epsilon)) / len(probs)
    print(f'Quantile level: {q_level}')
    qhat = np.quantile(probs, q_level, method='higher')
    print(f'Quantile value: {qhat}')
    
    # plot histogram and quantile
    plt.figure(figsize=(6, 2))
    plt.hist(probs, bins=60, edgecolor='k', linewidth=1)
    plt.axvline(
        x=qhat, linestyle='--', color='r', label='Quantile value'
    )
    plt.title(
        'Histogram of non-comformity scores in the calibration set'
    )
    plt.xlabel('Non-comformity score')
    plt.legend()
    plt.savefig('histogram.png')
     
    
if __name__ == "__main__":
    probs = []
    with open('answer.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            prob = line.split()[-1].split('\n')[0]
            probs.append(float(prob))
    calculate_threshold(probs)