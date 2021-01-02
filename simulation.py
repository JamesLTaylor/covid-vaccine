import json
import time
import numpy as np
from matplotlib.animation import FuncAnimation
import constants
from matplotlib import pyplot as plt
from numpy.random import uniform, normal
import datetime as dt


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
        self.daily_infections = []
        self.daily_r0 = []
        self.vaccinated = 0
        self.old_vaccinated = 0

    def save(self, filename):
        str = json.dumps(self.__dict__)
        with open(filename, 'w') as f:
            f.write(str)

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            d = json.load(f)
        self.active = d["active"]
        self.mild = d["mild"]
        self.severe = d["severe"]
        self.dead = d["dead"]
        self.recovered = d["recovered"]
        self.infections = d["infections"]
        self.deaths = d["deaths"]
        self.r0 = d["r0"]
        self.daily_infections = d["daily_infections"]
        self.daily_r0 = d["daily_r0"]


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
    def __init__(self, number: int, x: float, y: float, age: float):
        self.number = number
        self.x = x
        self.y = y
        self.patch = (int(x * patch_count), int(y * patch_count))
        patches[self.patch].add(self.number)
        self.infected = 0

        self.x0 = x
        self.y0 = y
        self.age = age
        self.is_fast_agent = False
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
        if self.is_fast_agent:
            lines = plt.plot(self.x, self.y, '*', ms=10)
        else:
            lines = plt.plot(self.x, self.y, 'o', ms=3)
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
    for i in range(patch_count):
        for j in range(patch_count):
            patches[(i, j)] = set()

    fast_agents = []
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
            agent = Agent(agent_counter, x, y, age)
            agents.append(agent)
            agent_locations[agent_counter][0] = x
            agent_locations[agent_counter][1] = y
            agent_counter += 1

    infected = 0
    target_infected = 1 + int(initial_infection * N)
    print(f"Ensuring that the simulation starts with {target_infected} infected agents in the top right quadrant")
    while infected < target_infected:
        agent_number = int(N * uniform())
        agent = agents[agent_number]
        if agent.state is constants.susceptible and agent.x > 0.5 and agent.y > 0.5:
            agent.infect(0)
            infected += 1

    target_fast = int(constants.fast_fraction*N)
    print(f"Ensuring that the simulation starts with {target_fast} fast agents")
    while len(fast_agents) < target_fast:
        if len(fast_agents) > 0:
            agent_number = int(N * uniform()) # reuse the prior agent number from infected to keep things interesting
        agent = agents[agent_number]
        if not agent.is_fast_agent:
            agent.is_fast_agent = True
            fast_agents.append(agent_number)

    return agents, agent_locations, fast_agents


def compact_totals():
    """
    Replace all r0 and infection records with a summary of number per day.
    :return:
    """
    infections = np.array(totals.infections)
    daily_infections = np.zeros(days)
    daily_r0 = np.zeros(days)
    r0_data = {}
    for agent in agents:
        if agent.state is constants.dead or agent.state is constants.immune:
            day = int(agent.time_in_state + constants.t_d)
            if day not in r0_data:
                r0_data[day] = [agent.infected]
            else:
                r0_data[day].append(agent.infected)
    for i in range(days):
        daily_infections[i] = len(np.where((infections >= (i - 1)) & (infections < i))[0])
        if i in r0_data:
            daily_r0[i] = np.mean(r0_data[i])
        else:
            daily_r0[i] = 0
    totals.infections = []
    totals.r0 = []
    totals.daily_infections = daily_infections.tolist()
    totals.daily_r0 = daily_r0.tolist()


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


def get_next_patient(fast_first, mode):
    if fast_first and totals.vaccinated < len(fast_agents):
        patient = fast_agents[totals.vaccinated]
    elif mode == 'old' or fast_first and totals.vaccinated >= len(fast_agents):
        patient = N - 1 - totals.old_vaccinated
        totals.old_vaccinated += 1
    elif mode == 'random' or fast_first and totals.vaccinated >= len(fast_agents):
        patient = int(N * uniform())
    else:
        raise Exception(f"Unknown mode {fast_first}, {mode}, {t}")
    if agents[patient].state == constants.vaccinated:
        print(f"{patient} already done")
        return get_next_patient(fast_first, mode)
    else:
        return patient


def vaccinate(t, fast_first, mode):
    """

    :param t:
    :param fast_first: Should we vaccinate the fast agents first?
    :param mode: 'old', 'random'
    :return:
    """
    if t <= constants.vaccine_delay:
        return
    target_total = int(t * constants.vaccine_rate * N)
    while totals.vaccinated < target_total:
        patient = get_next_patient(fast_first, mode)
        totals.vaccinated += 1
        agents[patient].state = constants.vaccinated
        print(f"{totals.vaccinated} vaccine given to patient {patient}, age = {agents[patient].age}, "
              f"is fast = {agents[patient].is_fast_agent}")


def update(frame_number, plot=False, vaccinate_fast_first=False, vaccination_mode=None):
    """
    Take on time step. Update infections and the progression of the disease in each agent.
    Optionally plot the agents.
    :param vaccination_mode: None, 'random' or 'old'
    :param vaccinate_fast_first: Should we vaccinate the fast agents first?
    :param frame_number:
    :param plot:
    :return:
    """
    t = frame_number * delta
    if frame_number % 50 == 0:
        print(t)
    for agent in agents:
        if agent.state is constants.dead:
            continue

        x = agent.x
        y = agent.y
        multiplier = 1.0
        if agent.is_fast_agent:
            multiplier = constants.fast_multiplier
        new_x = new_coord(x, multiplier * constants.speed)
        new_y = new_coord(y, multiplier * constants.speed)
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
            if uniform() < delta * beta:
                agent.infected += 1
                other_agent.infect(t)
            # dx = x - other_agent.x
            # dy = y - other_agent.y
            # dist = np.sqrt(dx * dx + dy * dy)
            # if dist < constants.r_c:
            #     if uniform() < delta * beta:
            #         other_agent.infect(t)
        agent.step(t)
        if vaccination_mode is not None:
            vaccinate(t, vaccinate_fast_first, vaccination_mode)

    if plot:
        infection_count = len(totals.infections)
        ax.set_title(f"{t:.1f} days. {infection_count} infections ({100 * len(totals.infections) / N:.1f}%). "
                     f"{totals.new_infections(t, 1)} last 24 hours. "
                     f"r0 = {totals.get_r0()}. "
                     f"{len(totals.deaths)} deaths")
        for agent in agents:
            # if agent.number % 10 == frame_number % 10:
            agent.draw_update(ax)


if __name__ == "__main__":
    # Initialize
    N = 1000
    t = 0
    delta = 1 / 10
    sdt = np.sqrt(delta)
    days = 25
    include_plot = False
    vaccinate_fast_first = False
    vaccination_mode = 'random'

    for sim_number in range(60):
        print(f"Simulation number: {sim_number}")
        print(f"Vaccination strategy = {vaccination_mode}. Vaccinate fast first = {vaccinate_fast_first}")
        patches = {}
        totals = Totals()
        r_c = constants.r_c_1000 / np.sqrt(N / 1000)  # r_c is normalized for 1000 agents
        patch_count = int(1 / r_c) + 1
        agents, agent_locations, fast_agents = init(N, constants.initial_infection)

        # Run the Simulation
        if include_plot:
            fig, ax = plot(agents)
            animation = FuncAnimation(fig, update, interval=200, save_count=days*10)
            animation.save('simulation_2000agents_250d_v2.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
            # plt.show()
        else:
            start = time.time()
            for i in range(days*10):
                update(i, include_plot, vaccinate_fast_first, vaccination_mode)
            print(f"{time.time() - start}")

        compact_totals()
        d = dt.datetime.now()
        # totals.save(f"./data_fast/{N}_{days}_{d.month}_{d.day}_{d.hour}_{d.minute}_{d.second}.json")

