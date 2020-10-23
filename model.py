import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


def main():
	data = pd.read_csv("data/Advertising.csv")
	data.head()
	data.drop(['Unnamed: 0'], axis=1)
	plt.figure(figsize=(16, 8))
	plt.scatter(
	    data['TV'],
	    data['sales'],
	    c='black'
	)
	plt.xlabel("Money spent on TV ads ($)")
	plt.ylabel("Sales ($)")
	plt.show()

	Xs = data.drop(['sales', 'Unnamed: 0'], axis=1)
	y = data['sales'].reshape(-1,1)
	reg = LinearRegression()
	reg.fit(Xs, y)
	print("The linear model is: Y = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".\
		format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))

	predictions = reg.predict(X)
	plt.figure(figsize=(16, 8))
	plt.scatter(
	    data['TV'],
	    data['sales'],
	    c='black'
	)
	plt.plot(
	    data['TV'],
	    predictions,
	    c='blue',
	    linewidth=2
	)
	plt.xlabel("Money spent on TV ads ($)")
	plt.ylabel("Sales ($)")
	plt.show()
	reg.score(Xs, y)

	X = np.column_stack((data['TV'], data['radio'], data['newspaper']))
	y = data['sales']

	X2 = sm.add_constant(X)
	est = sm.OLS(y, X2)
	est2 = est.fit()
	print(est2.summary())

if __name__ == '__main__':
	main()



