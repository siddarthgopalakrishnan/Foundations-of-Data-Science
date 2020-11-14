"""
1. Siddarth Gopalakrishnan - 2017B3A71379H
2. Rohan Maheshwari - 2017B4A70965H
3. Satvik Vuppala - 2017B4A71449H
"""

# Importing libraries
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

# Subplot to save images
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax1.set_xlabel("MEAN")
ax1.set_ylabel("Beta Distribution")
ax2.set_xlabel("MEAN")
ax2.set_ylabel("Beta Distribution")
ax1.set_title('Sequential')
ax2.set_title('Whole batch')

# Function to generate beta distribution
def generate_distribution(mu, alpha, beta):
    return (gamma(alpha+beta)/(gamma(alpha)*gamma(beta))) * ((mu)**(alpha-1) * (1-mu)**(beta-1))

# Let there be 50 ones and 110 zeros
m = 50
N = 160
print('Number of heads in our distribution = ' + str(m))
print('Number of tails in our distribution = ' + str(N-m))

# Generating dataset of 'm' heads(1) and 'N-m' tails(0)
o = np.ones(m)
print(o)
z = np.zeros(N-m)
print(z)
data = np.concatenate((o, z))
np.random.shuffle(data)  # Random shuffle to ensure random distribution of 1s and 0s
print(data)

# Mean of the dataset = m/N = 50/160 = 0.3125
mean = np.average(data)
print('Mean of the dataset = ' + str(mean))

# Declaring parameters of beta distribution to ensure prior has mean = 0.4
alpha = 2
beta = 3

# Sequential Learning Approach
x = ([i * 0.00001 for i in range(1,100000)]) # Defining interval

prior_value = 0
posterior_value = 0

for i in range(N):
	# Save image for each iteration
    y = ([generate_distribution(i * 0.00001, alpha, beta) for i in range(1,100000)])
    ax1.plot(x, y)
    #plt.savefig(str(i+1) + ".png")
    
    if i > 0:
        prior_value = posterior_value
    if data[i] == 1:
        alpha += 1
    else:
        beta += 1
    posterior_value = generate_distribution(mean, alpha, beta)
    print("Prior value = " + str(prior_value))
    print("Posterior value = " + str(posterior_value))

#plt.show()
#print("Mean, Alpha and Beta are =", mean, alpha, beta)
print("Posterior value by sequential approach = " + str(posterior_value))

# Entire Batch Approach
alpha = 2
beta = 3
alpha += m
beta += N-m
posterior_value = generate_distribution(mean, alpha, beta)
print("Mean, Alpha and Beta are =", mean, alpha, beta)
print("Posterior value by batch approach = " + str(posterior_value))
Y = ([generate_distribution(i * 0.00001, alpha, beta) for i in range(1,100000)])
ax2.plot(x, Y)
#plt.savefig("Entire batch.png")

plt.show()
#print("Final Posterior =", generate_distribution(0.3125, 52, 113))
#print("Final Posterior =", generate_distribution(0.5, 82, 83))
