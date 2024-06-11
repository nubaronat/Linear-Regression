import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns

plt.rcParams['figure.figsize'] = (12,8)

data = pd.read_csv('bike_sharing_data')
print(data.shape)

ax = sns.scatterplot(x="Population" , y = "Profit", data=data)
ax.set_title("Profit in $10000s vs City Population in 10000s")
plt.show()

#Compute the Cost J(theta)
def cost_functionh(X, y ,theta ):
    m = len(y)
    y_pred = X.dot(theta) #dot product
    error = (y_pred - y) ** 2
    return  1 / (2*m) * np.sum(error)

m = data.Population.values.size
X = np.append(np.ones((m,1)), data.Population.values.reshape(m,1), axis =1)
y = data.Profit.values.reshape(m , 1)
theta = np.zeros((2,1))
print(cost_functionh(X, y ,theta ))

#Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_pred = X.dot(theta)
        error = np.dot(X.transpose(), (y_pred - y))
        theta -= alpha * 1/m * error
        costs.append(cost_functionh(X,y,theta))
    return theta , costs

theta, costs = gradient_descent(X, y, theta, alpha=0.01, iterations=2000)
print("h(x) = {} + {}*1".format(str(round(theta[0,0],2)),
                                str(round(theta[1,0],2))))


