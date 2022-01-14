# Optimizing-Autoencoders-For-Dimensionality-Reduction-An-Evolutionary-Approach
Masterthesis University of Basel


The basic framework for the algorithm (frameworkDE) consisted of four files: functions.py, layers.py, model.py and optimizer.py. The algorithm itself was located in the module called optimizer.py, whereas all other modules provided classes required to implement differential evolution for autoencoder optimization.

The module model.py was used to create neural networks and perform the actions needed for the algorithm. 
In addition, a framework was built (frameworkADAM) to create AEs using Keras and then to optimize them with ADAM. The Experiments/CV folder contains files for cross-validation experiments, whereas the evaluation file (evaluation.py) is located in the result_datasetname folder.
