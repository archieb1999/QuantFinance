import numpy as np


class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, [1, self.iterations])

        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) +
                                       self.sigma * np.sqrt(self.T) * rand)

        option_data[:, 1] = stock_price - self.E

        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_simulation(self):
        option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, [1, self.iterations])

        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) +
                                       self.sigma * np.sqrt(self.T) * rand)

        option_data[:, 1] = self.E - stock_price

        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
        return np.exp(-1.0 * self.rf * self.T) * average


if __name__ == '__main__':
    model = OptionPricing(100, 100, 1, 0.05, 0.2, 1000000)
    print('Value of call option is $%.2f' % model.call_option_simulation())
    print('Value of put option is $%.2f' % model.put_option_simulation())
