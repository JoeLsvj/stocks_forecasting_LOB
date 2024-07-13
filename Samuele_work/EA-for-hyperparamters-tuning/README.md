# Explanation EA for hyper parameter tuning folder 

## Folders

**Past**
Past files, not still used 

**Data**
Data 

## Files

### Notebooks

**Operators for ga for layers**:
ES crossover and mutation operators for the tuning of the size and number of the layers

**ga for tuning 2**
A GA for selecting the best size and number of layers of the model 

**ga for tuning 1**
A GA for selecting the best activation function of the model

**ea for lr:**
A GA for founding the best lr of the optimizer of the NN

**change model paramters**
A NN for image recognition in which in the initialization you can set the parameters of the model 

### Py

**plot.py**
It contains functions to plot 

## The EA description 

### Super sum of the EA
- Kind = ($\mu$, $\lambda$) ES + elitism
- Fitness assessemnt = train loss of the model trained on the training data and tested on the validation data 
- Selection = inverse wheel selection + elitism
- Stopping criterion = number of fitness assessments    

### Complete description 
The chosen EA is an Evolutionary Strategy (ES) of the kind $(\mu,\lambda)$ with 1-elitism.
The initialization is standard and it produces $\lambda$ individuals.
The fitness assessment is the more expensive operation (indeed set the stopping criterion as the maximum number of fitness assessment done).
The selection is done by a inverse wheel selection + elitism. 
To be more specific, the selection is on the best individual (the one with the lowest fitness). 
Then, a probability distribution is compute over the fitness (the higher is the fitness, the lower is the probability).
Then, $\mu$ individual are selected. 
The generation of the $$ offsprings is created by crossover and mutation on the elite and the over $\mu$ individuals. 
Finally, the new population is given by the $\lambda - 1 $ offspings and the elite. 
The stopping criterion is the number of ffitness assessment done. 

