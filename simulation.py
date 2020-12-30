import time
from typing import List

import numpy as np
from matplotlib.animation import FuncAnimation

import constants
from matplotlib import pyplot as plt
from numpy.random import uniform, normal


class Totals:
    def __init__(self):
        self.active = 0
        self.mild = 0
        self.severe = 0
        self.dead = 0
        self.recovered = 0
        self.infections = []
        self.deaths = []
        self.r0 = []

    def new_infections(self, t, lag):
        temp = np.array(self.infections)
        count = len(temp[temp > (t-lag)])
        return count

    def infect(self, t):
        self.infections.append(t)
        self.active += 1

    def die(self, age, t, infected_before_die):
        self.deaths.append([t, age])
        self.dead += 1
        self.active -= 1
        self.r0.append(infected_before_die)

    def recover(self, age, t, infected_before_recover):
        self.recovered += 1
        self.active -= 1
        self.r0.append(infected_before_recover)

    def get_r0(self):
        if len(self.r0) < 10:
            return 0.0
        return np.mean(self.r0[-10:])


class Agent:
    def __init__(self, number: int, x: float, y: float, age: float, social_speed: float):
        self.number = number
        self.x = x
        self.y = y
        self.patch = (int(x * patch_count), int(y * patch_count))
        patches[self.patch].add(self.number)
        self.infected = 0

        self.x0 = x
        self.y0 = y
        self.age = age
        self.social_speed = social_speed
        self.state = constants.susceptible
        self.time_in_state = 0
        self.dot = None
        raw_mortality = np.interp(age, constants.death_rate[:, 0], constants.death_rate[:, 1])
        raw_mortality *= constants.total_mortality_multiplier  # mortality may be higher when we include untested people
        r = constants.proportion_mild
        scaled_mortality = raw_mortality / (r + (1 - r) * constants.sm)
        self.severe_mortality = constants.sm * scaled_mortality
        self.mild_mortality = scaled_mortality

    def infect(self, t):
        self.state = constants.exposed
        self.time_in_state = t
        totals.infect(t)

    def step(self, t: float):
        """
        Update state of agents based on their internal clocks
        :param t:
        :return:
        """
        elapsed = t - self.time_in_state
        if self.state == constants.exposed and elapsed > constants.t_l:
            if uniform() < constants.proportion_mild:
                self.state = constants.infected_mild
                totals.mild += 1
            else:
                self.state = constants.infected_severe
                totals.severe += 1
            self.time_in_state = t
        elif self.state == constants.infected_mild and elapsed > constants.t_d - constants.t_l:
            if uniform() < self.mild_mortality:
                self.state = constants.dead
                totals.die(self.age, t, self.infected)
            else:
                self.state = constants.immune
                totals.recover(self.age, t, self.infected)
        elif self.state == constants.infected_severe and elapsed > constants.t_d - constants.t_l:
            if uniform() < self.severe_mortality:
                self.state = constants.dead
                totals.die(self.age, t, self.infected)
            else:
                self.state = constants.immune
                totals.recover(self.age, t, self.infected)

    def color(self):
        """
        Oldest (90) gets 0.5, 0.5, 0.5, Youngest (0) gets (0.9. 0.9. 0.9)
        :return:
        """
        if self.state == 0:
            v = 0.9 - 0.4 * self.age / 90
            return v, v, v
        return constants.states[self.state][1]

    def plot(self, ax):
        """
        Plot points for first time
        :param ax:
        :return:
        """
        lines = plt.plot(self.x, self.y, 'o')
        self.dot = lines[0]
        self.dot.set_color(self.color())

    def draw_update(self, ax):
        """
        update point location
        :param ax:
        :return:
        """
        self.dot.set_color(self.color())
        self.dot.set_xdata(self.x)
        self.dot.set_ydata(self.y)

    def set_xy(self, new_x, new_y):
        """
        Se the x, y coordinates of the agent and update which patch they are found in.
        :param new_x:
        :param new_y:
        :return:
        """
        self.x = new_x
        self.y = new_y
        patch = (int(new_x * patch_count), int(new_y * patch_count))
        if self.patch == patch:
            return
        patches[self.patch].remove(self.number)
        patches[patch].add(self.number)
        self.patch = patch


def plot(agents):
    """
    Plot all the agents

    :param agents:
    :return:
    """
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    for agent in agents:
        agent.plot(ax)
    plt.show()
    return fig, ax


def init(N, initial_infection):
    fast_agents = set()
    agents = []
    agent_locations = np.zeros((N, 2))

    agent_counter = 0
    for row in range(constants.population_age.shape[0]):
        n = int(constants.population_age[row, 1] * N)
        if row == constants.population_age.shape[0] - 1:
            n = N - agent_counter
        for i in range(n):
            x = np.random.uniform()
            y = np.random.uniform()

            age = constants.population_age[row, 0]
            agent = Agent(agent_counter, x, y, age, constants.speed)
            if np.random.uniform() < initial_infection and x > 0.5 and y > 0.5:
                agent.infect(0)

            agents.append(agent)
            agent_locations[agent_counter][0] = x
            agent_locations[agent_counter][1] = y
            agent_counter += 1

    return agents, agent_locations, fast_agents


def new_coord(old_coord, speed):
    """
    Find a new x or y value by taking a brownian step and bouncing off the edges
    :param old_coord:
    :param speed:
    :return:
    """
    new = old_coord + speed * sdt * normal()
    if new <= 0:
        new = -new
    if new >= 1:
        new = 2 - new
    return new


def update(frame_number, plot=True):
    """
    Take on time step. Update infections and the progression of the disease in each agent.
    Optionally plot the agents.
    :param frame_number:
    :param plot:
    :return:
    """
    t = frame_number * dt
    print(t)
    for agent in agents:
        if agent.state is constants.dead:
            continue

        x = agent.x
        y = agent.y
        new_x = new_coord(x, constants.speed)
        new_y = new_coord(y, constants.speed)
        agent.set_xy(new_x, new_y)
        if agent.state is constants.exposed or agent.state is constants.infected_mild:
            beta = constants.beta_o
        elif agent.state is constants.infected_severe:
            beta = constants.beta_r
        else:
            continue

        # check new infections
        for other_agent_number in patches[agent.patch]:
            other_agent = agents[other_agent_number]
            if other_agent.number == agent.number:
                continue
            if other_agent.state is not constants.susceptible:
                continue
            if uniform() < dt * beta:
                agent.infected += 1
                other_agent.infect(t)
            # dx = x - other_agent.x
            # dy = y - other_agent.y
            # dist = np.sqrt(dx * dx + dy * dy)
            # if dist < constants.r_c:
            #     if uniform() < dt * beta:
            #         other_agent.infect(t)
        agent.step(t)

    if plot:
        infection_count = len(totals.infections)
        ax.set_title(f"{t:.1f} days. {infection_count} infections ({100 * len(totals.infections) / N:.1f}%). "
                     f"{totals.new_infections(t, 1)} last 24 hours. "
                     f"r0 = {totals.get_r0()}. "
                     f"{len(totals.deaths)} deaths")
        for agent in agents:
            # if agent.number % 10 == frame_number % 10:
            agent.draw_update(ax)


N = 2000
t = 0
dt = 1 / 10
sdt = np.sqrt(dt)

patches = {}
totals = Totals()
r_c = constants._r_c / np.sqrt(N/1000)  # r_c is normalized for 1000 agents
patch_count = int(1 / r_c) + 1
for i in range(patch_count):
    for j in range(patch_count):
        patches[(i, j)] = set()

agents, agent_locations, fast_agents = init(N, constants.initial_infection)

days = 100
include_plot = True
if include_plot:
    fig, ax = plot(agents)
    animation = FuncAnimation(fig, update, interval=200, save_count=days*10)
    animation.save('test1.mp4', fps=2, extra_args=['-vcodec', 'libx264'])
    # plt.show()
else:
    start = time.time()
    for i in range(days*10):
        update(i, include_plot)
    print(f"{time.time() - start}")


infections = np.array(totals.infections)
x = np.arange(days)
y = np.zeros(days)
for i in range(days):
    y[i] = len(np.where((infections >= (i-1)) & (infections < i))[0])

avg_window = 10
y_avg = np.convolve(totals.r0, np.ones(avg_window)/avg_window, mode='valid')
plt.plot(y_avg)
