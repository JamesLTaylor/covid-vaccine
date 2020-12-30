import numpy as np

# https://www.worldometers.info/coronavirus/coronavirus-age-sex-demographics/
death_rate = np.array([[0, 0],
                       [10, 0.002],
                       [20, 0.002],
                       [30, 0.002],
                       [40, 0.004],
                       [50, 0.013],
                       [60, 0.036],
                       [70, 0.08],
                       [80, 0.148]])

# https://en.wikipedia.org/wiki/Demographics_of_South_Africa
population_age = np.array([[2, 0.11],
                           [7, 0.093],
                           [12, 0.089],
                           [17, 0.097],
                           [22, 0.104],
                           [27, 0.098],
                           [32, 0.078],
                           [37, 0.067],
                           [42, 0.057],
                           [47, 0.051],
                           [52, 0.043],
                           [57, 0.035],
                           [62, 0.027],
                           [67, 0.019],
                           [72, 0.014],
                           [77, 0.009],
                           [82, 0.006],
                           [87, 0.005]])


susceptible = 0
exposed = 1
infected_mild = 2
infected_severe = 3
dead = 4
immune = 5
states = {susceptible: ["Susceptible", "Gray"],
          exposed: ["Exposed", "fuchsia"],
          infected_mild: ["Infected (mild)", "Orange"],
          infected_severe: ["Infected (severe)", "Red"],
          dead: ["Dead", "Black"],
          immune: ["Recovered/vaccine", "Green"]}

t_d = 14  # disease duration = infection to dead/better
t_l = 5  # infection to symptom


proportion_mild = 0.95  #(r) proportion of symptom free infections
sm = 5  # mortality multiplier: severe / mild
# require that r * lambda + (1-r) * sm * lambda = total mortality

dt = 1 / 100

initial_infection = 0.004
speed = 0.01  # per day
_r_c = 0.1  # social distance at which person gets infected. Need such that average R is 2.5
beta_r = 0.04
beta_o = beta_r / 3  # infection rate from mild symptom agents


total_mortality_multiplier = 2.0
