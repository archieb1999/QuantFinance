# Quantitative Finance Models

This repository contains a collection of quantitative finance models implemented in Python. These models were developed during my undergraduate studies at IIT ISM, when I was exploring the field of quantitative finance. The repository includes implementations of various pricing models, risk management techniques, and portfolio optimization methods.

## Table of Contents
1. Monte Carlo Simulation for Stock Price Prediction
2. Ornstein-Uhlenbeck Process
3. Vasicek Interest Rate Model
4. Value at Risk (VaR)
5. Value at Risk (VaR) with Monte Carlo Simulation
6. Monte Carlo Simulation for Option Pricing
7. Black-Scholes Option Pricing Model
8. Bond Pricing using Vasicek Model and Monte Carlo Simulation
9. Capital Asset Pricing Model (CAPM)
10. Markowitz Portfolio Optimization

### 1. Monte Carlo Simulation for Stock Price Prediction
The `MonteCarlo.py` script implements a Monte Carlo simulation to predict future stock prices. The model assumes that stock returns follow a normal distribution and uses the geometric Brownian motion to simulate price paths. The script generates multiple simulations and calculates the average stock price across all simulations.

Example code:
```python
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
    return simulation_data['mean'].tail(1)
```

### 2. Ornstein-Uhlenbeck Process
The `OrnsteinUhlenbeck.py` script demonstrates the simulation of an Ornstein-Uhlenbeck process. This process is commonly used to model mean-reverting phenomena, such as interest rates or commodity prices. The process is governed by the following stochastic differential equation:

$dx(t) = \theta(\mu - x(t))dt + \sigma dW(t)$

where $x(t)$ is the value of the process at time $t$, $\theta$ is the speed of mean reversion, $\mu$ is the long-term mean, $\sigma$ is the volatility, and $W(t)$ is a standard Brownian motion.

Example code:
```python
def generate_process(dt=0.1, theta=1.2, mu=0.5, sigma=0.3, n=10000):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
    return x
```

### 3. Vasicek Interest Rate Model
The Vasicek model, implemented in `Vasicek.py`, is a stochastic model for interest rates. It assumes that the interest rate follows an Ornstein-Uhlenbeck process, which is a mean-reverting process. The model is described by the following stochastic differential equation:

$dr(t) = \kappa(\theta - r(t))dt + \sigma dW(t)$

where $r(t)$ is the interest rate at time $t$, $\kappa$ is the speed of mean reversion, $\theta$ is the long-term mean interest rate, $\sigma$ is the volatility of the interest rate, and $W(t)$ is a standard Brownian motion.

Example code:
```python
def vasicek_model(r0, kappa, theta, sigma, T=1, N=1000):
    dt = T/float(N)
    t = np.linspace(0,T, N+1)
    rates = [r0]
    for _ in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.sqrt(dt)*np.random.normal()
        rates.append(rates[-1] + dr)
    return t, rates
```

### 4. Value at Risk (VaR)
Value at Risk (VaR) is a widely used risk management metric that quantifies the potential loss for a given confidence level and time horizon. The `VaR.py` script calculates the VaR of a stock position using historical stock price data. It assumes that stock returns follow a normal distribution and estimates the VaR based on the mean and standard deviation of the returns.

Example code:
```python
def calculate_var(position, c, mu, sigma, n):
    var = position * (-mu*n - sigma * np.sqrt(n) * norm.ppf(1 - c))
    return var
```

### 5. Value at Risk (VaR) with Monte Carlo Simulation
The `VaRmontecarlo.py` script extends the VaR calculation by using Monte Carlo simulation. Instead of assuming a normal distribution for stock returns, it generates multiple simulations of the stock price path based on historical data. The VaR is then estimated as the specified percentile of the simulated portfolio values.

Example code:
```python
def simulation(self):
    rand = np.random.normal(0, 1, [1, self.iterations])
    stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) +
                                  self.sigma * np.sqrt(self.n) * rand)
    stock_price = np.sort(stock_price)
    percentile = np.percentile(stock_price, (1 - self.c)*100)
    return self.S - percentile
```

### 6. Monte Carlo Simulation for Option Pricing
The `montecarlo2.py` script uses Monte Carlo simulation to price European call and put options. The model assumes that the underlying stock price follows a geometric Brownian motion. By generating multiple simulations of the stock price path, the script estimates the option prices as the average of the discounted payoffs across all simulations.

Example code:
```python
def call_option_simulation(self):
    option_data = np.zeros([self.iterations, 2])
    rand = np.random.normal(0, 1, [1, self.iterations])
    stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) +
                                   self.sigma * np.sqrt(self.T) * rand)
    option_data[:, 1] = stock_price - self.E
    average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
    return np.exp(-1.0 * self.rf * self.T) * average
```

### 7. Black-Scholes Option Pricing Model
The Black-Scholes model, implemented in `BlackScholes.py`, is a widely used model for pricing European call and put options. The model assumes that the underlying stock price follows a geometric Brownian motion and derives an analytical formula for the option price based on the stock price, strike price, time to expiration, risk-free interest rate, and volatility of the stock.

The Black-Scholes formula for a call option is given by:

$C(S, t) = S \cdot N(d_1) - K \cdot e^{-r(T-t)} \cdot N(d_2)$

where:
- $S$ is the current stock price
- $K$ is the strike price
- $r$ is the risk-free interest rate
- $T-t$ is the time to expiration
- $\sigma$ is the volatility of the stock
- $N(\cdot)$ is the cumulative distribution function of the standard normal distribution
- $d_1 = \frac{\ln(\frac{S}{K}) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}}$
- $d_2 = d_1 - \sigma\sqrt{T-t}$

The formula for a put option is similar, with some modifications to the payoff structure.

Example code:
```python
def call_option_price(S, E, T, rf, sigma):
    d1 = (log(S/E) + (rf + sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*stats.norm.cdf(d1)-E*exp(-rf*T)*stats.norm.cdf(d2)
```

### 8. Bond Pricing using Vasicek Model and Monte Carlo Simulation
The `BondPricingVasicek.py` script combines the Vasicek interest rate model with Monte Carlo simulation to price bonds. The Vasicek model is used to simulate the evolution of interest rates, and the bond price is calculated as the expected value of the discounted cash flows over all simulated interest rate paths.

Example code:
```python
def monte_carlo_simulation(x, r0, kappa, theta, sigma, T=1.0):
    dt = T / float(NUM_OF_POINTS)
    result = []
    for _ in range(NUM_OF_SIMULATIONS):
        rates = [r0]
        for _ in range(NUM_OF_POINTS):
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        result.append(rates)
    simulation_data = pd.DataFrame(result).T
    integral_sum = simulation_data.sum() * dt
    present_integral_sum = np.exp(-integral_sum)
    bond_price = x * np.mean(present_integral_sum)
    return bond_price
```

### 9. Capital Asset Pricing Model (CAPM)
The Capital Asset Pricing Model (CAPM) is a widely used model in finance that describes the relationship between the expected return and risk of an asset. The `CAPM.py` script implements the CAPM using historical stock and market data. It calculates the beta coefficient of a stock, which measures its sensitivity to market movements, and estimates the expected return based on the risk-free rate and the market risk premium.

Example code:
```python
def calculate_beta(self):
    covariance_matrix = np.cov(self.data['s_returns'], self.data['m_returns'])
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta
```

### 10. Markowitz Portfolio Optimization
The Markowitz portfolio optimization model, implemented in `markowitz.py`, aims to find the optimal allocation of assets in a portfolio that maximizes the expected return for a given level of risk. The model uses historical stock price data to estimate the expected returns and covariance matrix of the assets. It then solves an optimization problem to determine the optimal weights of the assets in the portfolio.

Example code:
```python
def optimize_portfolio(weights, returns):
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
```

These models provide a foundation for understanding and implementing various concepts in quantitative finance, including asset pricing, risk management, and portfolio optimization. Each model has its assumptions and limitations, and it is important to consider them when applying these models in practice.

Please note that the implementations in this repository are for educational purposes and may not reflect the complexities and nuances of real-world financial markets. It is recommended to use these models as a starting point and to further enhance them based on specific requirements and market conditions.

The ordering of the models in the table of contents has been adjusted to reflect a progression from simpler to more complex models, starting with Monte Carlo simulations for stock price prediction and option pricing, moving on to interest rate models like Vasicek and Ornstein-Uhlenbeck, and then covering risk management techniques such as Value at Risk (VaR). The Black-Scholes model for option pricing is included as a more advanced pricing model. Finally, the Capital Asset Pricing Model (CAPM) and Markowitz portfolio optimization represent higher-level concepts in asset pricing and portfolio management.
