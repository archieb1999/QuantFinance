import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_OF_SIMULATIONS = 1000


def stock_monte_carlo(S0, mu, sigma, N=252):
    result = []

    for _ in range(NUM_OF_SIMULATIONS):

        prices = [S0]

        for _ in range(N):
            stock_price = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * np.random.normal())
            prices.append(stock_price)

        result.append(prices)

    simulation_data = pd.DataFrame(result)
    simulation_data = simulation_data.T
    simulation_data['mean'] = simulation_data.mean(axis=1)

    plt.plot(simulation_data['mean'])
    plt.show()

    print('Prediction for future stock price: $%.2f' % simulation_data['mean'].tail(1))


if __name__ == '__main__':
    stock_monte_carlo(50, 0.0002, 0.01)
