# Schelling and The Pursuit of Happyness
## Geospatial Analytics - A.A. 2022/23

This repository contains the source code of the final project for the course "Geospatial Analytics" at the University of Pisa.

The question we want to answer is the following: can we learn relocation strategies in a Schelling simulation that improve convergence time?

The code that asks this question can be primarly found in the Jupyter Notebook `learn_to_relocate.ipynb`.

### Phase 1: MESA-based simulation

Write and run a Mesa script implementing a mild random policy for relocation. Collect the complete traces of the simulation, extracting at least the sequence of relocations performed along the simulation.

### Phase 2: build a training set

Analyzing the simulation traces, create a dataset that contains a row for each relocation happened, composed of `<agent-feat, start-feat, end-feat, happiness_duration>`, where:
- `agent-feat`: this is a set of computed features describing the agent. Basic features should include the number of time steps where the agent relocated. The student can propose others.
- `start-feat`: this is a set of computed features describing the context of the location where the agent was before relocating. Basic features should include the coordinates in the grid and the number of similar agents in the neighborhood. The student can propose others.
- `end-feat`: same as start-feat, but for the destination location.
- `happiness_duration`: a number describing for how many simulation steps, starting from the relocation, the agent remained in the destination location. This is equivalent to say, for how many time steps the agent remained happy - before relocating again.

### Phase 3: learn to relocate

Discretize the field happiness_duration into two classes: "short" and "long", based on a duration threshold that you decide through analysis of the distribution of values. Use the dataset you obtain to train a classification model that predicts the happiness duration class based on the other fields (excepted happiness_duration, of course). Try with multiple classification methods and follow the usual process to select the best combination of model + parameters. If the training set is too small, run step 1 multiple times (on different initial configurations), to enlarge it. We call the model “ML-Schelling”.

Now, modify the Mesa script in such a way that when an agent has to relocate it uses the classification model to choose the destination. More precisely, we compute all the features adopted in the training phase (namely: `agent-feat`, `start-feat` and `end-feat`) for all pairs `(o, d)`, where `o` is fixed and corresponds to the actual location of the agent, and `d` is any of the remaining locations. Clearly, the components `agent-feat` and `start-feat` are always the same, whereas `end-feat` changes with each possible destination. We apply the model to each pair and select the one that predicts "long" happiness with the highest confidence/probability.


### Phase 4: test the model

Run a set of simulations on different initial conditions using both the "mild random" and the ML Schelling policies, and evaluate, on each simulation setting, how many iterations are needed with the two methods to converge and how long, on average, does happiness of agents last. Does ML give an advantage?