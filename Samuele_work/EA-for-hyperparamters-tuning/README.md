# Explanation EA for hyper parameter tuning folder 

## Folders

**Past**
Past files, not still used 

**Data**
Data 

## Files

### Notebooks

**ga for tuning all**
A ES for tune lr, af and structure of the NN (currently the notebook is just a copy of the *ga for tuning 3*, i have to update the cells)

**ga for tuning 4**
A GA for selecting the best size and number of layers of the model (gene = real numbers) with refined cycle and operators. 

**ga for tuning 3**
A GA for selecting the best size and number of layers of the model (gene = real numbers)

**Operators for ga for layers**:
ES crossover and mutation operators for the tuning of the size and number of the layers

**ga for tuning 2**
A GA for selecting the best size and number of layers of the model (gene = class = solution)

**ga for tuning 1**
A GA for selecting the best activation function of the model

**ea for lr:**
A GA for founding the best lr of the optimizer of the NN

**change model parameters**
A NN for image recognition in which in the initialization you can set the parameters of the model 

### Py

**plot.py**
It contains functions to plot 

## Bibliography
**EA for hyper paramter tuning in NN** \
https://link.springer.com/article/10.1007/s10690-023-09412-z

Idea for the hyper paramter tuning of our NN

**ES understanding**
https://link.springer.com/book/10.1007/978-3-662-05094-1

Choice of the ES $(\mu,\lambda)$ against the ES $\mu +\lambda$.

**ES understading**
https://direct.mit.edu/evco/article/1/1/1/1092/An-Overview-of-Evolutionary-Algorithms-for

Choice of the $\mu$ and $\lambda$ parameters in the ES.

## Deatiled explanation of algorithm used the notebooks

### ea for lr
A GA for founding the best activation function for the model. 

The spaces are
- The genotype space is: $\R$
- The solution space is: $\R$

The evolutionary cycle is:
- Initialization (init)
- Repeat
  - Fitness assessment (FA)
    - On: current population (POP) 
  - Selection 
    - On: POP
    - kind: wheel selection + 1 elite 
  - Generate offsprings 
    - Operators on: selection 
    - Operators: Crossover (CX) and Mutation (MUT) with a probability $p$
  - Update POP
    - POP = offsprings   
  
  The CX is intermediate_recombination.

  The MUT is adding noise from a uniform distrution on a constant interval. 

### ga for tuning 1
A GA for founding the best lr of the optimizer of the NN.

The spaces are
- The genotype space is: $\N^q$
- The solution space is: List of activation function 

The evolutionary cycle is:
- Init
- Repeat
  - FA
    - On: current POP
  - Selection 
    - On: POP
    - kind: wheel selection + 1 elite 
  - Generate offsprings 
    - Operators on: selection 
    - Operators: CX and MUT with a probability $p$
  - Update POP
    - POP = offsprings   
  
The CX is deterministic_best_class.

The MUT is exctracting activation function class randomly. 

### ga for tuning 2
A GA for selecting the best size and number of layers of the model (gene = class = solution).

The spaces are
- The genotype space is: $2^{\N}$
- The solution space is: $2^{\N}$

The evolutionary cycle is:
- Init
- Repeat
  - FA
    - On: current POP
  - Selection 
    - On: POP
    - kind: wheel selection + 1 elite 
  - Generate offsprings 
    - Operators on: selection 
    - Operators: CX or MUT (with probability $[0.5,0.5]$ )
  - Update POP
    - POP = offsprings   
  
The CX is selected between
- single_point_crossover
- two_point_crossover
    
The MUT is selected across
- random_resetting_mutation
- increment_decrement_mutation
- swap_mutation
- shrink_mutation
- grow_mutation

### Thoughts
**IMP** \
The problem is that the best individual is every generation an individual with only one layer, and after the test, it is not the best structure of the dl.
Indeed, the structure [256,128,64,32] is better then [512,512] that is better then [512]. 
In my opinion, there are two possible reasons why the ES prefers individual with one layer:
- The wheel selection is too small
- The num of epochs per fitness assessment are few 
  
Regarding the num of epochs, this means that there are few epochs to assess and individual. 
The problem of the larger NN is that they require longer time to become stronger then the other. 

**Change the number of epochs per fitness assessement** \
I changed the number of epochs per fitness assessemnet. 
Consequently, i had to drecease the size of the population to computational time reasons. 
It seems now the model prefer NN with more layers. 
This confirms what I was expecting.

On the other hand there is premature convergence.

There are more possibility to handle this problem.
- Increase the population size 
- Use a ES $\mu + \lambda$ instead of a $(\mu, \lambda)$ + torunament selection.
  
At the moment, increasing the population is impossbile due to the computational power. 

**Wheel selection digression** \
The wheel selection pro is the maintainance of a strong diversity, the con is the problems with the fitness scaling and a lower convergence.
Let suppose the fitness of the individual is different but about 0.009, 0.00091, 0.0092 ecc.. the probability distribution becomes almost a uniform distribution, that is not our goal in this contex. 

**ES $\mu + \lambda$ and torunament selection** \
In the ES $\mu + \lambda$ the population increases generation per generation. 
Therefore, the wheel selection may become less effective beacuse it picks some bad individual sometimes. 

So, in this case i prefer using the tournament selection. 
However, the size of the tournament must be adjusted generation per generation, beacuse the size of the population grows over time. 

There are different possible percentages:
- 5-10 \% : low selection pressure (pro: high diversity, slower convergence)
- 10-15 \% : moderate selection pressure (pro: balance btw exploration and exploitation, risk of premature convergence)
- 15-20 \% : high selection pressure (pro: faster convergence, high risk of premature convergence)



### ga for tuning 3
A GA for selecting the best size and number of layers of the model (gene = real numbers).

The spaces are
- The genotype space is: $2^{\R}$
- The solution space is: $2^{\N}$

The evolutionary cycle is:
- Init
- Repeat
  - FA
    - On: current POP
  - Selection 
    - On: POP
    - kind: tournament selection
  - Generate offsprings 
    - Operators on: selection 
    - Operators: CX and grow/shrink MUT with a probability $p$ and Integer MUT
  - Update POP
    - POP = POP + offsprings   
  
The CX is single_point_crossover_fitness_based_with_E
    
The MUT is 
- shrink mutation (with probaility $p/2$)
- grow mutation (with probaility $p/2$)
- integer_mutation_preprocess



### ga for tuning 4
A GA for selecting the best size and number of layers of the model (gene = real numbers) refined in the cycle and in the operators. 

The spaces are
- The genotype space is: $2^{\R}$
- The solution space is: $2^{\N}$

The evolutionary cycle is:
- Init ($\mu$ individuals)
- FA 
  - On: Initial POP ($\mu$)
- Repeat
  - Generate offsprings 
    - Operators on: parents (taken one by one from the population with tournament selection)
    - Operators: CX or MUT on parents 
    - How many: $\lambda$
  - FA
    - On: offsprings
  - Selection 
    - On: offsprings
    - kind: tournament selection
    - How many: $\mu$
  - Update POP
    - POP = offsprings 
  
The CX is k_singlePointCrossover_fitnessBased.
    
The MUT is adaptive_gaussian_mutation.




