import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def plot_progression(days, totals, N):
    avg_window = 10
    daily_infections_avg = np.convolve(totals.daily_infections, np.ones(avg_window) / avg_window, mode='valid')
    x_vals = np.arange(days)
    plt.figure()
    plt.plot(x_vals[avg_window - 1:], daily_infections_avg)
    for death in totals.deaths:
        day = int(death[0])
        y = daily_infections_avg[day - avg_window]
        plt.plot(day, y, 'ok')
        plt.text(day + 2.5, y - 0.1, int(death[1]))
    plt.title("Daily new infections. (10day rolling average)")
    plt.xlabel("days")
    plt.ylabel("new infections")
    plt.figure()
    plt.plot(x_vals, 100 * np.cumsum(totals.daily_infections) / N)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    plt.title("Total infections.")
    plt.xlabel("days")
    plt.ylabel("% of population infected")
    plt.figure()
    avg_window = 21
    daily_r0_avg = np.convolve(totals.daily_r0, np.ones(avg_window) / avg_window, mode='valid')
    plt.plot(x_vals[avg_window - 1:], daily_r0_avg)
    plt.hlines(1, x_vals[avg_window - 1], x_vals[-1], 'k', linestyles='dotted')
    plt.title("Realized $R_0$. 21 day rolling average")
    plt.xlabel("days")
    plt.ylabel("average realized $R_0$")

