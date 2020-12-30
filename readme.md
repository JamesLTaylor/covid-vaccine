# Covid Vaccine Strategy for South Africa

Inspired by the work [Grauer2020], we attempt to answer the question of how to optimally administer a limited
supply of covid-19 vaccines in order to minimize deaths.

There are two extra features that we believe could be interesting to add to the Grauer method and investigate.
 
 * Mortality increases with age
 * There could be highly connected individuals like shops workers, taxi drivers, security personal etc.
 
There are several strategies that can be considered for administering the vaccine for example.
 
 1. Give to those most at risk of succumbing to the disease
 2. Give to areas that have the highest infection rate
 3. Give uniformly to the population
 4. Give to people who come into contact with the most others people


strategies 1. and 4. require the above proposed modifications to the original model.

## Method

We model N agents who move through "social distance" space following Brownian motions. Infected agents can 
infect susceptible agents in the same patch of space. Agents start uniformly distributed in space and the 
patches in which infections can take place are squares with size such that the expected number of 
agents in each square is a constant. The initial agents are simulated with ages based on the 2011 South
 African census 
as reported on [wikipedia](https://en.wikipedia.org/wiki/Demographics_of_South_Africa#Age_and_sex_distribution).

## Results 1

Parameters:

|Parameter|Value|
|---|---|
|N|2000|
|in patch|10|
|||


Click on the following image to watch a youtube video of an example simulation:

[![Simulation](http://img.youtube.com/vi/7wdFTtFDvIw/0.jpg)](http://www.youtube.com/watch?v=7wdFTtFDvIw)

This simulation runs for 250 days and produces the following population dynamics: 

![daily infections](./images/v1 daily infections.png "daily infections")

Black dots represent deaths and the ages of the simulated agents that died. 

![total infections](./images/v1 total infections.png "total infections")

![realized r0](./images/v1 realized r0.png "realized r0")


## Issues


The current implementation is too slow to run with more than several thousand agents. Because the mortality 
is around 2% that means that even when 20% of a population of 10000 has had Covid-19, one would only expect
40 deaths. There is a lot of numerical noise in such a small number so it will be hard to measure the impact
of the vaccination strategies.
  

## Data

The mortality rates are based on the age of the agent and the severity of the infection.

### Age based mortality rates


### Age distribution in population



## References   



[Grauer2020]: Grauer, J., Löwen, H. & Liebchen, B. Strategic spatiotemporal vaccine distribution increases 
the survival rate in an infectious disease like Covid-19. Sci Rep 10, 21594 (2020). 
[https://doi.org/10.1038/s41598-020-78447-3]