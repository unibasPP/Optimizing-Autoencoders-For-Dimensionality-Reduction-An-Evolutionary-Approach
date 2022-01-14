

# Import necessary packages
#--------------------------
import numpy as np
import numpy.random as rd
from frameworkDE.model import ANN


#-------------------------------------------
# Differential Evolution for Neural Networks
#-------------------------------------------      

class DENN:
    # initialie the class
    def __init__(self, popSize, n_nodes, activation, loss, niter, strat='rand', F=None, CR=None, jDE=True, k=None, p=None, w_scheme=None, bounds=None):
        # Initialize Standard DE parameters
        self.popSize    = popSize
        self.niter      = niter
        self.strat      = strat
        self.population = []
        self.fitValues  = []
        self.bestAE     = None
        self.bestAEVali = None
        self.fitOpt     = 9999
        self.fitVali    = 0
        self.fitTest    = 0
        self.endVali    = 0
        self.optReport  = []
        self.valiReport = []
        self.iE         = 0
        self.iE_test    = 0
        self.bounds     = bounds        
        # jDE adaptive strategy
        self.jDE        = jDE        
        self.CR         = self.init_jDE_CR() if jDE else CR
        self.F          = self.init_jDE_F() if jDE else F
        self.Fm1        = (rd.rand(self.popSize) * 0.5) if strat=='MERGE' else None        
        # DEGL strategy parameters
        self.neighbors  = self.neighborhood(k) if k is not None else k
        self.w_scheme   = w_scheme # different adaption schemes for w
        self.w          = []
        self.wt         = 0
        # current-to-best with external archive parameters 
        self.p          = p
        self.A          = [] # with external archive
        # Initialize model parameter
        self.n_nodes    = n_nodes
        self.activation = activation
        self.loss       = loss



    def init_Population(self, X):
        #initialize Population of agents
        for indx in range(self.popSize):
            #initialize ANN
            agent = ANN(self.n_nodes, self.activation)
            self.population.append(agent)
            # Evaluate ANN
            obj_f = self.fitness(X, agent)
            self.fitValues.append(obj_f)
            # Self-Adaptive weight factor initialization if needed
            if self.w_scheme == 'SAW':
                self.w.append(rd.rand())
            # select elitest
            if obj_f < self.fitOpt:
                self.fitOpt = obj_f
                self.bestAE = agent
                self.iE     = (len(self.population)-1)
        # number of agents to check
        n_agents = len(self.population)
        print('Initialized a population size: %d'%n_agents)
        

    def fitness(self, X, agent):
        # calc network output
        out = agent.forward(X)
        # calc the loss
        return self.loss.calc(out, X)
    
    
    def agent_to_vec(self, agent):
        # flatten the entire ANN to a vector
        return np.concatenate([np.ravel(layer.weights.copy(), order='F') for layer in agent.model])
    
    
    def vec_to_agent(self, vec, old_agent):
        # set starting index
        from_i = 0
        to_i   = 0
        new_layers = []
        # create matrices according to old agent
        for layer in old_agent.model:
            to_i = from_i + np.prod(layer.weights.size)
            new_weights = vec[from_i:to_i].copy()
            new_W       = new_weights.reshape(layer.weights.shape, order='F')
            new_layers.append(new_W)
            from_i = to_i
        # return new layers
        return new_layers


    # define neighbors
    def neighborhood(self, radius):
        if radius <= ((self.popSize-2)/2) and radius > 0:
            # define neighborhood indices for each individual in the population
            neighborss = []
            for NP in range(self.popSize):
                # NP is the center of the radius and popSize is the max index
                x = [i % (self.popSize) for i in range(NP-radius, NP+radius+1) if (i != NP)]
                neighborss.append(x)
            #return
            return list(np.array(neighborss))
        else:
            print("Radius must be between 0 and (popSize-1)/2")
     
            
    # find individual in neighborhood with best fitness
    def findBestNeighbor(self, idx):
        # take index of lowest (min prob) fitness value in neighborh.
        fitVal = np.array(self.fitValues)
        ind = np.argmin(fitVal[self.neighbors[idx]])
        # return best neighbor
        return self.neighbors[idx][ind]
    
    
    # create neighboorhood index for local donor vector creation
    def locBestIndex(self, idx):
        # find best indexes in the neighborhood of each indiv. in population
        bestNei = self.findBestNeighbor(idx)
        # list to store 2 random vecs from the neigh. diff from indiv./best
        # index of individual (target vector)
        candidate = [i for i in self.neighbors[idx] if (i != bestNei)]
        rando = list(rd.choice(candidate, 2, replace=False))
        # return indexes for creation of donor vector
        return [bestNei] + rando
 
    
    # global index 
    def globBestIndex(self, idx):
        # find global best index in current iteration in population
        i_G = np.argmin(self.fitValues) 
        candidate = [i for i in range(self.popSize) if (i != i_G) and (i != idx)]
        rando = list(rd.choice(candidate, 2, replace=False)) 
        # reuturn indexes for global best strategy
        return [i_G] + rando  

    
    def pBestIndex(self, idx):
        # find the p%Best index in current iteration in population
        n = int(np.ceil(self.p * self.popSize))
        fit = self.fitValues.copy()
        # index for best to worst according to fitness and chose one of p%best
        i_pBest = np.argsort(fit)[rd.randint(0, n)]
        # random candidate 
        candidate = [i for i in range(self.popSize) if (i != i_pBest) and (i != idx)]
        rando = list(rd.choice(candidate, 1, replace=False))
        # with archive A
        candidate1 = [i for i in range(self.popSize + len(self.A)) if (i != i_pBest) and (i != idx) and (i != rando)]
        a = list(rd.choice(candidate1, 1, replace=False))
        return [i_pBest] + rando + a
    
    
    def mergeIndex(self, idx):
        # find the p%Best index in current iteration in population
        n = int(np.ceil(self.p * self.popSize))
        fit = self.fitValues.copy()        
        # index for best to worst according to fitness and chose one of p%best
        i_pBest = np.argsort(fit)[rd.randint(0, n)]
        # chose one of the p%worst
        i_pWorst = np.argsort(fit)[::-1][rd.randint(0, n)]
        # chose one from the middle (popSize-2n)
        i_middle = np.argsort(fit)[n:(self.popSize-n)][rd.randint(0, (self.popSize-2*n))]
        # return candidate 
        return [i_pBest, i_pWorst, i_middle]     
    
    
    # function to chose random indexes for mutation 
    def randIndex(self, idx):
        # create possible indexes without individuum        
        candidate = [i for i in range(self.popSize) if (i != idx)]
        # return 3 random mutualy excl. indices also != individual (index)
        return list(rd.choice(candidate, 3, replace=False))
 

    # jDE adaptive strategy of parameters CR and F
    #------
    def init_jDE_F(self):
        # initial F 0.5
        return (0.1 + rd.rand(self.popSize) * (0.9))
    
    
    def init_jDE_CR(self):
        # initial CR 0.9
        return (rd.rand(self.popSize))
    
    
    def trial_jDE(self, idx):
        # initialize a random F and CR value for each indiviual in PopSize
        # according to jDE
        if rd.rand() < 0.1:
            F1 = (0.1 + rd.rand()*(0.9))
        else:
            F1 = self.F[idx]
        # CR
        if rd.rand() < 0.1:
            CR1 = (rd.rand())
        else:
            CR1 = self.CR[idx]
        # return trial values
        return F1, CR1
    
    
    def merge_jDE(self, idx):
        # initialize a random F and CR value for each indiviual in PopSize
        # according to jDE
        if rd.rand() < 0.1:
            F1 = (rd.rand()*0.5)
        else:
            F1 = self.Fm1[idx]
        # return trial values
        return F1    


    def encoder(self, X, test=False):
        # get index of latent + 1 for slicing operation is already done in forward
        latent = np.argmin(self.n_nodes[1:])
        # get encoder output
        if test:
            return self.bestAEVali.forward(X, i_stop_layer=latent)
        else:
            return self.bestAE.forward(X, i_stop_layer=latent)
    
    def decoder(self, X_encoded, test=False):
        # index for decoder start
        i_decoder = np.argmin(self.n_nodes[1:]) + 1
        # get decoder output
        if test:
            return self.bestAEVali.forward(X_encoded, i_start_layer=i_decoder)
        else:
            return self.bestAE.forward(X_encoded, i_start_layer=i_decoder)
           
    def predict(self, X, test=False):
        if test:
            return self.bestAEVali.forward(X) 
        else:
            return self.bestAE.forward(X) 
    
    def evaluate(self, X, idx=None, test=False):       
        # Perform forward pass
        output = self.predict(X, test)
        # calc the loss
        loss = self.loss.calc(output, X)
        # only if not in iterations
        if idx is None:
            # Print a report
            print(f'Validation 'f'loss: {loss:0.3f}')
        else:
            return loss


    def mutationXO(self, agent_x, Ft, CRt, idx, iters):
        # Rand strategy
        #---------------
        if self.strat == 'rand':
            # chose 3 random indices (mutually excl. and diff. from individual)
            abc_idx = self.randIndex(idx)
            # chose agents from population with that index
            agent_a, agent_b, agent_c = [self.population[i] for i in abc_idx]
            # vectorize all agents
            vec_x = self.agent_to_vec(agent_x)
            vec_a = self.agent_to_vec(agent_a)
            vec_b = self.agent_to_vec(agent_b)            
            vec_c = self.agent_to_vec(agent_c) 
            # create new trial solution
            vec_xt = vec_x.copy()
            sieve  = rd.rand(len(vec_xt)) <= CRt 
            vec_xt[sieve]  = vec_a[sieve] + Ft*(vec_b[sieve] - vec_c[sieve])
            #return trial solution
            return vec_xt.copy()
        # current to best
        #----------------
        elif self.strat == 'current-to-best':
            # define global strategy
            abc_idx = self.globBestIndex(idx)
            # chose global agents from population with that index
            best, r1, r2 = [self.population[i] for i in abc_idx]
            
            # vectorize all agents
            vec_x      = self.agent_to_vec(agent_x)
            # global
            vec_best = self.agent_to_vec(best)
            vec_r1   = self.agent_to_vec(r1)            
            vec_r2   = self.agent_to_vec(r2)  
            
            # create global trial vector
            vec_xt  = vec_x.copy()
            sieve   = rd.rand(len(vec_xt)) <= CRt 
            vec_xt[sieve]  = vec_x[sieve] + Ft*(vec_best[sieve] - vec_x[sieve]) + Ft*(vec_r1[sieve] - vec_r2[sieve])

            # return trial vector
            return vec_xt.copy()         
        # DEGL strategy
        #---------------        
        elif self.strat == 'DEGL':
            # define local and global strategy
            abc_idx_L = self.locBestIndex(idx)
            abc_idx_G = self.globBestIndex(idx)
            # set the weight factor w for DEGL according to scheme
            if self.w_scheme == 'LI': # linear increment
                w = (iters/self.niter)
            elif self.w_scheme == 'EI': # exponential increment
                w = (np.exp((iters*np.log(2))/self.niter)-1)
            elif self.w_scheme == 'randw':
                w = (rd.rand())
            elif self.w_scheme == 'SAW': # self adaptive w
                # take global best indices and apply to weights
                best_w, r1_w, r2_w = [self.w[i] for i in abc_idx_G]
                # individual weight
                i_w = self.w[idx]
                # muation (no crossover for w)
                w = i_w + Ft*(best_w - i_w) + Ft*(r1_w - r2_w)
                # clip the value between [0.05, 0.95]
                w = np.clip(w, 0.05, 0.95)
                self.wt = w
            
            # chose local agents from population with that index
            best_L, r1_L, r2_L = [self.population[i] for i in abc_idx_L]
            # chose global agents from population with that index
            best_G, r1_G, r2_G = [self.population[i] for i in abc_idx_G]
            
            # vectorize all agents
            vec_x      = self.agent_to_vec(agent_x)
            # local
            vec_best_L = self.agent_to_vec(best_L)
            vec_r1_L   = self.agent_to_vec(r1_L)            
            vec_r2_L   = self.agent_to_vec(r2_L)
            # global
            vec_best_G = self.agent_to_vec(best_G)
            vec_r1_G   = self.agent_to_vec(r1_G)            
            vec_r2_G   = self.agent_to_vec(r2_G)  
            
            # create local and global donor vector
            vec_xd_L  = vec_x + Ft*(vec_best_L - vec_x) + Ft*(vec_r1_L - vec_r2_L)
            vec_xd_G  = vec_x + Ft*(vec_best_G - vec_x) + Ft*(vec_r1_G - vec_r2_G)
            # combine local and global donor vector 
            vec_xd    = w*vec_xd_G + (1-w)*vec_xd_L
            # use binomial crossover
            vec_xt        = vec_x.copy()
            sieve         = rd.rand(len(vec_xt)) <= CRt
            vec_xt[sieve] = vec_xd[sieve]
            # return trial vector
            return vec_xt.copy()
        # current-to-pbest with external archive strategy
        elif self.strat == 'current-to-pbest':
            # define pBest strategy
            abc_idx = self.pBestIndex(idx)
            ab_idx  = abc_idx[:-1]
            c_idx   = abc_idx[-1]
            # because of archive
            if c_idx < self.popSize:
                # chose agents from population with that index
                pBest, r1, rArch = [self.population[i] for i in abc_idx]
                vec_rArch = self.agent_to_vec(rArch)                
            else:
                c_idx = (c_idx - self.popSize)
                # chose agents agents from population and archive
                pBest, r1 = [self.population[i] for i in ab_idx]
                rArch = self.A[c_idx] 
                vec_rArch = self.agent_to_vec(rArch)
                
            # vectorize all agents
            vec_x      = self.agent_to_vec(agent_x)
            # agent to vector
            vec_pBest = self.agent_to_vec(pBest)
            vec_r1    = self.agent_to_vec(r1)            
            # create new trial solution
            vec_xt = vec_x.copy()
            sieve  = rd.rand(len(vec_xt)) <= CRt 
            vec_xt[sieve]  = vec_x[sieve] + Ft*(vec_pBest[sieve] - vec_x[sieve]) + Ft*(vec_r1[sieve] - vec_rArch[sieve])            
            #return trial solution
            return vec_xt.copy()
        # MERGE strategy
        #---------------        
        elif self.strat == 'MERGE':
            Ft, F1 = Ft
            # define local and global strategy
            abc_idx_L  = self.locBestIndex(idx)
            abc_idx_G   = self.mergeIndex(idx)
            # set the weight factor w for DEGL according to scheme
            if self.w_scheme == 'LI': # linear increment
                w = (iters/self.niter)
            elif self.w_scheme == 'EI': # exponential increment
                w = (np.exp((iters*np.log(2))/self.niter)-1)
            elif self.w_scheme == 'randw':
                w = (rd.rand())
            elif self.w_scheme == 'SAW': # self adaptive w
                # individual weight
                i_w = self.w[idx]            
                # take global best indices and apply to weights
                best_w, worst_w = [self.w[i] for i in abc_idx_G[:-1]]
                # muation (no crossover for w)
                w = i_w + Ft*(best_w - i_w) + F1*(i_w - worst_w)
                # clip the value between [0.05, 0.95]
                w = np.clip(w, 0.05, 0.95)
                self.wt = w
            
            # chose local agents from population with that index
            best_L, r1_L, r2_L = [self.population[i] for i in abc_idx_L]                
            # chose global agents from population with that index
            best_G, worst_G, middle_G = [self.population[i] for i in abc_idx_G]  
            
            # vectorize all agents
            vec_x      = self.agent_to_vec(agent_x)            
            # local
            vec_best_L = self.agent_to_vec(best_L)
            vec_r1_L   = self.agent_to_vec(r1_L)            
            vec_r2_L   = self.agent_to_vec(r2_L)
            # global
            vec_best_G  = self.agent_to_vec(best_G)
            vec_worst_G = self.agent_to_vec(worst_G) 
            vec_middle_G = self.agent_to_vec(middle_G) 
            
            # create local and global donor vector
            vec_xd_L  = vec_x + Ft*(vec_best_L - vec_x) + Ft*(vec_r1_L - vec_r2_L)
            vec_xd_G  = vec_middle_G + Ft*(vec_best_G - vec_middle_G) + F1*(vec_middle_G - vec_worst_G) + F1*(rd.randn(len(vec_x)))
            # combine local and global donor vector 
            vec_xd    = w*vec_xd_G + (1-w)*vec_xd_L
            # use binomial crossover
            vec_xt        = vec_x.copy()
            sieve         = rd.rand(len(vec_xt)) <= CRt
            vec_xt[sieve] = vec_xd[sieve]
            # return trial vector
            return vec_xt.copy()   
            
            

    def evolution(self, X, validation=None, test=None):
        
        # Initialize population
        self.init_Population(X)
        
        # iterate over generations
        for iters in range(self.niter):
            improvement=0                                       
            # iterate over population
            for idx in range(len(self.population)):
                # take current individual and its fitness value
                agent_x    = self.population[idx]
                fitness_x  = self.fitValues[idx]
                # set F, CR if necessary according to adaptive strategy jDE
                if self.jDE and not self.strat=='MERGE':
                    Ft, CRt = self.trial_jDE(idx)
                elif self.jDE and self.strat=='MERGE':
                    F1, CRt = self.trial_jDE(idx)
                    F2      = self.merge_jDE(idx)  
                    Ft      = [F1, F2]
                else:
                    Ft, CRt = self.F, self.CR
                
                # Mutation and crossover 
                #------------------------
                # do mutation and crossver and return trial vec
                vec_xt = self.mutationXO(agent_x, Ft, CRt, idx, iters)
                # clip according to bounds
                if isinstance(self.bounds, list):
                    l, u   = self.bounds
                    vec_xt = np.clip(vec_xt, l, u)
                
                # Evaluate trial solution
                aWeights = self.vec_to_agent(vec_xt, agent_x)
                #create new trial agent
                agent_xt    = ANN(self.n_nodes, self.activation, weights=aWeights)
                fitness_xt  = self.fitness(X, agent_xt)   
                
                # Minimize (replace if lower reconstruction error)
                if fitness_xt <= fitness_x:# <= to overcome flat landscapes
                    self.population[idx] = agent_xt
                    self.fitValues[idx]  = fitness_xt
                    improvement += 1
                    # take better parameters in jDE
                    if self.jDE and not self.strat=='MERGE':
                        self.F[idx]  = Ft
                        self.CR[idx] = CRt
                    else:
                        F1, F2        = Ft
                        self.F[idx]   = F1
                        self.Fm1[idx] = F2
                        self.CR[idx]  = CRt 
                    # take better w value if SAW
                    if self.w_scheme=='SAW':
                        self.w[idx] = self.wt
                    # if strategy is current-to-pbest with external archive
                    if self.strat == 'current-to-pbest':
                        # update archive but never larger than popSize
                        if len(self.A) < self.popSize:
                            self.A.append(agent_x)
                        else:
                            ix = rd.randint(0, self.popSize)
                            self.A[ix] = agent_x 
                    # maybe new best solution
                    if fitness_xt <= self.fitOpt:
                        self.fitOpt = fitness_xt
                        self.bestAE = agent_xt
                        self.iE     = idx            

            # calculate validation error if not None
            if validation is not None:
                self.fitVali = self.evaluate(validation, idx)
                self.valiReport.append(self.fitVali)

            # apend current error optimum to report
            self.optReport.append(self.fitOpt)
            # print report            
# =============================================================================
#             print('DE Iteration %d: improved %d/%d agents (best %.6f / validation %.6f)'%\
#                     (iters, improvement, len(self.population), self.fitOpt, self.fitVali))
# =============================================================================
        # which AE is best on valiset
        if validation is not None:
            self.endVali = [self.fitness(validation, self.population[i]) for i in range(self.popSize)]
        # calculate error with unseen data (test set) with best of validation set
        if test is not None:
            self.iE_test = np.argmin(self.endVali)
            self.bestAEVali = self.population[self.iE_test]
            self.fitTest = self.evaluate(test, 1, True)
        # return best AE
        return self.bestAE if test is None else self.bestAEVali




