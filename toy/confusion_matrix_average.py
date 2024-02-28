import numpy as np

# Example confusion matrices
cm1 = np.array([[5, 2], [3, 4]])
cm2 = np.array([[4, 1], [2, 5]])
cm3 = np.array([[6, 3], [1, 6]])
cm4 = np.array([[5, 2], [2, 7]])
cm5 = np.array([[7, 1], [2, 6]])

# Summing all the confusion matrices
cm_sum = cm1 + cm2 + cm3 + cm4 + cm5

# Calculating the average
cm_average = cm_sum / 5

print("Sum of confusion matrices:\n", cm_sum)
print("Average confusion matrix:\n", cm_average)