# Quantitative Finance Models

This repository contains a collection of quantitative finance models implemented in Python. These models were studied and implemented during my undergraduate years, when I was exploring the field of quantitative finance out of personal interest alongside my formal studies in geophysics. The repository includes implementations of various pricing models, risk management techniques, and portfolio optimization methods that have been widely used in the field of quantitative finance since the mid-20th century. Many of these models have undergone extensions and enhancements in the late 20th century and recent years.

## Table of Contents

1. [Monte Carlo Simulation for Stock Prices](#monte-carlo-simulation-for-stock-prices)
2. [Ornstein-Uhlenbeck Process](#ornstein-uhlenbeck-process)
3. [Vasicek Interest Rate Model](#vasicek-interest-rate-model)
4. [Value at Risk (VaR)](#value-at-risk-var)
5. [Monte Carlo Simulation for VaR](#monte-carlo-simulation-for-var)
6. [Monte Carlo Option Pricing](#monte-carlo-option-pricing)
7. [Black-Scholes Option Pricing](#black-scholes-option-pricing)
8. [Bond Pricing using Vasicek Model](#bond-pricing-using-vasicek-model)
9. [Capital Asset Pricing Model (CAPM)](#capital-asset-pricing-model-capm)
10. [Markowitz Portfolio Optimization](#markowitz-portfolio-optimization)

## Monte Carlo Simulation for Stock Prices

### Background
Monte Carlo simulations are used to model the probability of different outcomes in a process that is influenced by random variables. This method is widely used in finance to simulate the future price of stocks or assets. The basic idea is to simulate a large number of potential future stock price paths and then analyze the distribution of the simulated outcomes.

### Mathematical Formulation
The stock price $S_t$ at time $t$ can be modeled using the geometric Brownian motion:
$S_t = S_0 \exp\left( \left( \mu - \frac{\sigma^2}{2} \right)t + \sigma W_t \right)$
where:
- $S_0$ is the initial stock price
- $\mu$ is the expected return
- $\sigma$ is the volatility
- $W_t$ is a Wiener process (or standard Brownian motion)

### Example Usage
```python
if __name__ == '__main__':
    S0 = 100      # Initial stock price
    mu = 0.1      # Expected annual return 
    sigma = 0.2   # Annual volatility
    T = 1         # Time horizon in years
    N = 252       # Number of trading days
    num_simulations = 1000  # Number of simulations

    # Perform Monte Carlo simulation
    simulation_data = stock_monte_carlo(S0, mu, sigma, N, num_simulations)

    # Plot the simulated stock price paths
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_data)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Monte Carlo Simulation of Stock Prices')
    plt.grid(True)
    plt.show()

    # Print the predicted stock price at the end of the time horizon
    predicted_price = simulation_data.iloc[-1]['mean']
    print(f'Predicted stock price at the end of {T} year(s): ${predicted_price:.2f}')
```

## Ornstein-Uhlenbeck Process

### Background
The Ornstein-Uhlenbeck process is a mean-reverting stochastic process used to model interest rates, currency exchange rates, and commodity prices.

### Mathematical Formulation
The Ornstein-Uhlenbeck process is described by the stochastic differential equation:
$dx_t = \theta (\mu - x_t) dt + \sigma dW_t$
where:
- $x_t$ is the variable of interest at time $t$
- $\theta$ is the speed of reversion
- $\mu$ is the long-term mean
- $\sigma$ is the volatility
- $W_t$ is a Wiener process

### Example Usage
```python
if __name__ == '__main__':
    theta = 1.0   # Speed of mean reversion
    mu = 0.5      # Long-term mean
    sigma = 0.1   # Volatility
    T = 1.0       # Time horizon
    N = 1000      # Number of time steps

    # Generate the Ornstein-Uhlenbeck process
    ou_process = generate_process(T/N, theta, mu, sigma, N)

    # Plot the generated process
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, T, N), ou_process)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Ornstein-Uhlenbeck Process')
    plt.grid(True)
    plt.show()
```

## Vasicek Interest Rate Model

### Background
The Vasicek model is used to model the evolution of interest rates. It assumes that interest rates follow a mean-reverting stochastic process.

### Mathematical Formulation
The Vasicek model is described by the stochastic differential equation:
$dr_t = \kappa (\theta - r_t) dt + \sigma dW_t$
where:
- $r_t$ is the interest rate at time $t$
- $\kappa$ is the speed of reversion
- $\theta$ is the long-term mean rate
- $\sigma$ is the volatility
- $W_t$ is a Wiener process

### Example Usage
```python
if __name__ == '__main__':
    r0 = 0.05     # Initial interest rate
    kappa = 0.5   # Speed of mean reversion
    theta = 0.08  # Long-term mean interest rate
    sigma = 0.02  # Volatility of interest rate
    T = 2.0       # Time horizon in years
    N = 1000      # Number of time steps

    # Generate the interest rate paths using the Vasicek model
    t, r = vasicek_model(r0, kappa, theta, sigma, T, N)

    # Plot the generated interest rate paths
    plt.figure(figsize=(10, 6))
    plt.plot(t, r)
    plt.xlabel('Time')
    plt.ylabel('Interest Rate')
    plt.title('Vasicek Interest Rate Model')
    plt.grid(True)
    plt.show()
```

## Value at Risk (VaR)

### Background
Value at Risk (VaR) is a measure of the risk of loss for investments. It estimates how much a set of investments might lose, given normal market conditions, in a set time period such as a day.

### Mathematical Formulation
VaR can be calculated using the historical method, variance-covariance method, or Monte Carlo simulation. For a given confidence level $c$, the VaR is given by:
$\text{VaR} = \mu + \sigma \Phi^{-1}(1-c)$
where:
- $\mu$ is the mean of the portfolio returns
- $\sigma$ is the standard deviation of the portfolio returns
- $\Phi^{-1}$ is the inverse of the cumulative distribution function of the normal distribution

### Example Usage
```python
if __name__ == '__main__':
    start_date = '2010-01-01'
    end_date = '2023-06-01'
    stock = 'AAPL'  # Stock symbol
    position = 1e6  # Position size
    confidence_level = 0.95  # Confidence level for VaR

    # Download historical stock data
    stock_data = download_data(stock, start_date, end_date)
    
    # Calculate daily returns
    stock_data['returns'] = stock_data['Adj Close'].pct_change()
    stock_data = stock_data[1:]

    # Calculate VaR
    mu = stock_data['returns'].mean()
    sigma = stock_data['returns'].std()
    var = calculate_var(position, confidence_level, mu, sigma, 1)

    print(f'Value at Risk (VaR) for {stock}: ${var:.2f}')
```

## Monte Carlo Simulation for VaR

### Background
Monte Carlo simulation for VaR involves generating a large number of possible future price paths for the portfolio based on the statistical properties of the portfolio returns.

### Mathematical Formulation
Given a confidence level $c$, the VaR using Monte Carlo simulation can be computed by simulating $N$ price paths and calculating the $(1-c) \times 100 \%$ percentile of the distribution of simulated returns.

### Example Usage
```python
if __name__ == '__main__':
    position = 1e6         # Position size
    confidence_level = 0.95  # Confidence level for VaR
    num_simulations = 100000  # Number of simulations
    
    # Define the stock parameters
    stock = 'AAPL'  # Stock symbol
    start_date = '2010-01-01'
    end_date = '2023-06-01'

    # Download historical stock data
    stock_data = download_data(stock, start_date, end_date)
    stock_data['returns'] = stock_data['Adj Close'].pct_change()
    stock_data = stock_data[1:]

    # Calculate parameters for Monte Carlo simulation
    mu = stock_data['returns'].mean()
    sigma = stock_data['returns'].std()

    # Create an instance of ValueAtRiskMonteCarlo
    model = ValueAtRiskMonteCarlo(position, mu, sigma, confidence_level, 1, num_simulations)

    # Perform the Monte Carlo simulation
    var = model.simulation()

    print(f'Value at Risk (VaR) for {stock} using Monte Carlo Simulation: ${var:.2f}')
```

## Monte Carlo Option Pricing

### Background
Monte Carlo methods can also be used to price European call and put options by simulating a large number of potential future paths for the underlying asset and averaging the discounted payoff of the option.

### Mathematical Formulation
The price of a call option using Monte Carlo simulation is given by:
$C = e^{-rT} \frac{1}{N} \sum_{i=1}^N \max(S_T^{(i)} - E, 0)$
where:
- $S_T^{(i)}$ is the simulated price of the asset at maturity
- $E$ is the exercise price
- $r$ is the risk-free rate
- $T$ is the time to maturity

### Example Usage
```python
if __name__ == '__main__':
    S0 = 100      # Initial stock price
    E = 110       # Strike price
    T = 1         # Time to maturity in years
    rf = 0.05     # Risk-free rate
    sigma = 0.2   # Annual volatility
    num_simulations = 100000  # Number of simulations

    # Create an instance of OptionPricing
    model = OptionPricing(S0, E, T, rf, sigma, num_simulations)

    # Calculate the call and put option prices using Monte Carlo simulation
    call_price = model.call_option_simulation()
    put_price = model.put_option_simulation()

    print(f'Call option price: ${call_price:.2f}')
    print(f'Put option price: ${put_price:.2f}')
```

## Black-Scholes Option Pricing

### Background
The Black-Scholes model is used to calculate the theoretical price of European call and put options. It assumes that the price of the underlying asset follows a geometric Brownian motion with constant drift and volatility.

### Mathematical Formulation
The price of a European call option $C$ and put option $P$ can be calculated using the following formulas:
$C = S_0 N(d_1) - E e^{-rT} N(d_2)$
$P = E e^{-rT} N(-d_2) - S_0 N(-d_1)$
where:
$d_1 = \frac{\ln(S_0 / E) + (r + \sigma^2 / 2)T}{\sigma \sqrt{T}}$
$d_2 = d_1 - \sigma \sqrt{T}$

### Example Usage
```python
if __name__ == '__main__':
    S0 = 100      # Initial stock price
    E = 110       # Strike price
    T = 1         # Time to maturity in years
    rf = 0.05     # Risk-free rate
    sigma = 0.2   # Annual volatility

    # Calculate the call and put option prices using the Black-Scholes model
    call_price = call_option_price(S0, E, T, rf, sigma)
    put_price = put_option_price(S0, E, T, rf, sigma)

    print(f'Call option price: ${call_price:.2f}')
    print(f'Put option price: ${put_price:.2f}')
```

## Bond Pricing using Vasicek Model

### Background
The Vasicek model can be used to price bonds by simulating future interest rate paths and discounting the bond's cash flows accordingly.

### Mathematical Formulation
The price of a zero-coupon bond can be derived from the Vasicek model and is given by:
$P(t,T) = A(t,T) e^{-B(t,T)r_t}$
where $A(t,T)$ and $B(t,T)$ are functions of the model parameters and the time to maturity $T-t$.

### Example Usage
```python
if __name__ == '__main__':
    r0 = 0.05     # Initial interest rate
    kappa = 0.5   # Speed of mean reversion
    theta = 0.08  # Long-term mean interest rate
    sigma = 0.02  # Volatility of interest rate
    T = 2.0       # Time to maturity in years
    N = 1000      # Number of time steps
    F = 1000      # Face value of the bond
    num_simulations = 10000  # Number of simulations

    # Calculate the bond price using Monte Carlo simulation
    bond_price = monte_carlo_simulation(F, r0, kappa, theta, sigma, T, N, num_simulations)

    print(f'Bond price: ${bond_price:.2f}')
```

## Capital Asset Pricing Model (CAPM)

### Background
The Capital Asset Pricing Model (CAPM) describes the relationship between systematic risk and expected return for assets, particularly stocks. It is used throughout finance for pricing risky securities and generating expected returns for assets.

### Mathematical Formulation
The expected return of a security $R_i$ is given by:
$R_i = R_f + \beta_i (R_m - R_f)$
where:
- $R_f$ is the risk-free rate
- $\beta_i$ is the beta of the security
- $R_m$ is the expected return of the market

### Example Usage
```python
if __name__ == '__main__':
    stock = 'AAPL'  # Stock symbol
    market = '^GSPC'  # Market index symbol (S&P 500)
    start_date = '2010-01-01'
    end_date = '2023-06-01'

    # Create an instance of CAPM
    capm = CAPM([stock, market], start_date, end_date)

    # Initialize the CAPM model
    capm.initialize()

    # Calculate the beta of the stock
    beta = capm.calculate_beta()
    print(f'Beta of {stock}: {beta:.2f}')

    # Perform regression analysis
    capm.regression()
```

## Markowitz Portfolio Optimization

### Background
Markowitz's Modern Portfolio Theory (MPT) helps in the selection of investment portfolios that maximize returns for a given level of risk by diversifying investments. It quantifies the benefits of diversification and provides a framework for constructing optimal portfolios.

### Mathematical Formulation
The objective is to minimize the portfolio variance:
$\sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$
subject to:
$\sum_{i=1}^n w_i = 1$
where:
- $\mathbf{w}$ is the vector of portfolio weights
- $\mathbf{\Sigma}$ is the covariance matrix of asset returns

The expected return of the portfolio is given by:
$\mu_p = \mathbf{w}^T \mathbf{\mu}$
where:
- $\mathbf{\mu}$ is the vector of expected returns of the assets

The optimization problem can be solved using quadratic programming to find the optimal portfolio weights that minimize the portfolio variance for a given expected return.

### Example Usage
```python
if __name__ == '__main__':
    start_date = '2010-01-01'
    end_date = '2023-06-01'
    
    # Download historical stock data
    stock_data = download_data()

    # Plot the stock prices
    show_data(stock_data)

    # Calculate daily log returns
    log_daily_returns = calculate_return(stock_data)

    # Generate random portfolios
    portfolio_weights, portfolio_returns, portfolio_volatilities = generate_portfolios(log_daily_returns)

    # Plot the efficient frontier
    show_portfolios(portfolio_returns, portfolio_volatilities)

    # Optimize the portfolio
    optimum = optimize_portfolio(portfolio_weights, log_daily_returns)

    # Print the optimal portfolio
    print_optimal_portfolio(optimum, log_daily_returns)

    # Plot the optimal portfolio on the efficient frontier
    show_optimal_portfolio(optimum, log_daily_returns, portfolio_returns, portfolio_volatilities)
```

These models provide a foundation for understanding and implementing various concepts in quantitative finance, including asset pricing, risk management, and portfolio optimization. Each model has its assumptions and limitations, and it is important to consider them when applying these models in practice.

Please note that the implementations in this repository are for educational purposes and may not reflect the complexities and nuances of real-world financial markets. It is recommended to use these models as a starting point and to further enhance them based on specific requirements and market conditions.

The ordering of the models in the table of contents has been adjusted to reflect a progression from simpler to more complex models, starting with Monte Carlo simulations for stock price prediction and option pricing, moving on to interest rate models like Vasicek and Ornstein-Uhlenbeck, and then covering risk management techniques such as Value at Risk (VaR). The Black-Scholes model for option pricing is included as a more advanced pricing model. Finally, the Capital Asset Pricing Model (CAPM) and Markowitz portfolio optimization represent higher-level concepts in asset pricing and portfolio management.
