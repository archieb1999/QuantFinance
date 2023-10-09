import numpy as np
import matplotlib.pyplot as plt


def generate_process(dt=0.1, theta=1.2, mu=0.5, sigma=0.3, n=10000):
    x = np.zeros(n)

    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))

    return x


def plot_process(x):
    plt.plot(x)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Ornstein-Uhlenbeck Process')
    plt.show()


if __name__ == '__main__':
    data = generate_process(n=10000, theta=1, sigma=0.05)
    plot_process(data)
