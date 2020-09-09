import numpy as np
import seaborn as sns
import random

# normal distribution
x = np.random.normal(loc=500.0, scale=1.0, size=10000)

sample_mean = []

# bootstrap sample from normal distribution
for i in range(40):
    y = random.sample(x.tolist(), 5)
    avg = np.mean(y)
    sample_mean.append(avg)

# compare mean of normal distribution with mean from bootstrapped
# samples
actual_mean = np.mean(x)
bootstrapped_mean = np.mean(sample_mean)
to_print = "actual mean: {} computed mean: {}".format(actual_mean, bootstrapped_mean)
print(to_print)