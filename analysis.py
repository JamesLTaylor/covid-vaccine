import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import simulation


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


def plot_envelope(paths_to_use, title, ylabel, y_axis_percent):
    plt.figure()
    plt.title(title)
    plt.xlabel("days")
    plt.ylabel(ylabel)
    median = np.percentile(paths_to_use, 50, axis=0)
    up_10 = np.percentile(paths_to_use, 90, axis=0)
    up_q = np.percentile(paths_to_use, 75, axis=0)
    low_10 = np.percentile(paths_to_use, 10, axis=0)
    low_q = np.percentile(paths_to_use, 25, axis=0)
    plt.fill_between(x_vals[avg_window - 1:], low_q, up_q, alpha=0.5, color="black")
    plt.fill_between(x_vals[avg_window - 1:], low_10, up_10, alpha=0.3, color="blue")
    plt.plot(x_vals[avg_window - 1:], median, 'k')
    for i in range(paths_to_use.shape[0]):
        plt.plot(x_vals[avg_window - 1:], paths_to_use[i, :], 'gray', alpha=0.2)

    custom_lines = [Line2D([0], [0], color="blue", lw=6, alpha=0.5),
                    Line2D([0], [0], color="blue", lw=6, alpha=0.3),
                    Line2D([0], [0], color="gray", lw=2, alpha=0.2)]

    plt.legend(custom_lines, ['25-75% Range', '10-90% Range', 'Individual'])
    if y_axis_percent:
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())


days = 250
N = 6000
avg_window = 10
x_vals = np.arange(days)
path = "./data_fast"
filenames = list(os.listdir(path))

all_deaths = []
paths = np.zeros((len(filenames), 250-avg_window+1))
r0_paths = np.zeros((len(filenames), 250-avg_window+1))
total_paths = np.zeros((len(filenames), 250-avg_window+1))
for i in range(len(filenames)):
    totals = simulation.Totals()
    totals.load_from_file(os.path.join(path, filenames[i]))
    daily_infections_avg = np.convolve(totals.daily_infections, np.ones(avg_window) / avg_window, mode='valid')
    paths[i, :] = daily_infections_avg

    daily_r0_avg = np.convolve(totals.daily_r0, np.ones(avg_window) / avg_window, mode='valid')
    r0_paths[i, :] = daily_r0_avg

    total_infected = 100 * np.cumsum(daily_infections_avg) / N
    total_paths[i, :] = total_infected

    all_deaths += [row[1] for row in totals.deaths]

plot_envelope(paths, "Daily new infections. (10day rolling average)", "new infections", False)
plot_envelope(r0_paths, "Average $R_0$. (10day rolling average)", "Realized $R_0$", False)
plot_envelope(total_paths, "Total infections.", "% of population infected$", True)

plt.figure()
plt.hist(all_deaths, np.arange(0, 100, 5))
plt.title("Distribution of simulated covid-19 deaths by age")
plt.xlabel("Age (years)")
plt.ylabel("Total from all simulations")