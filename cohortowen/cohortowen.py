import numpy as np
import itertools
import tqdm_pathos

def all_permutations_generator(n_vars):
    return np.array(list(itertools.permutations(range(n_vars))))
               
class CohortShapley():
    def __init__(self, model, similarity, subject_indices, data, func=np.average, y=None,
            parallel=0, pid=0, data_weight=None, permutations=None,
            mc_num = None,
            verbose=1):
        self.model = model
        self.data = data
        self.n_predictors = data.shape[1]
        self.n_observations = data.shape[0]
        self.similarity_function = similarity
        self.subject_indices = subject_indices
        self.func = func
        self.parallel = parallel
        self.pid = pid
        self.verbose=verbose
        #add in weighting todo
        #self.data_weight = data_weight
        
        self.permutations = permutations
        self.permutation_flag = isinstance(self.permutations, np.ndarray)

        if  mc_num is not None:
            permutations = np.zeros((mc_num, self.n_predictors), dtype=int)
            for k in range(mc_num):
                permutations[k] = np.random.permutation(self.n_predictors)
            self.permutations = permutations
        elif self.permutations is None:
            self.permutations = np.array(list(all_permutations_generator(self.n_predictors)))
        
        self.n_permutations = self.permutations.shape[0]
        
        if y is None:
            self.printlog("use given model to predict y.")
            self.y = model(data)
        else:
            self.y = y
            self.printlog("use given y values instead of model prediction.")

    def printlog(self, str):
        if self.verbose>0:
            print(str)

    def save(self, prefix):
        np.save(prefix + '.cs.npy', self.shapley_values)

    def load(self, prefix):
        self.shapley_values = np.load(prefix + '.cs.npy')
        
    def calculate_shapley_value_one(self,subject_index):
        similarity_table = self.similarity_function(self.data,subject_index)
        shaps = np.zeros([self.n_predictors])
        for i in range(self.n_permutations):
            mask = np.ones([self.n_observations])
            cohort_val = self.func(np.extract(mask,self.y))
            for j in range(self.n_predictors):
                mask = np.logical_and(similarity_table[:,self.permutations[i][j]],mask)
                new_val = self.func(np.extract(mask,self.y))
                shaps[self.permutations[i][j]] += (new_val - cohort_val)
                cohort_val = new_val
        return shaps/self.n_permutations


    def compute_cohort_shapley(self):
        
        if self.permutation_flag:
            self.printlog("compute Shapley values based on permutations")
            
        if len(self.subject_indices) == 1:
            self.shapley_values = self.calculate_shapley_value_one(self.subject_indices)
        elif self.parallel > 0:
            self.printlog("parallel processing with {0} processes".format(self.parallel))
            shapleys = tqdm_pathos.map(self.calculate_shapley_value_one,self.subject_indices,n_cpus=self.parallel)
            self.shapley_values = np.vstack(shapleys)
        else:
            self.shapley_values = np.array([self.calculate_shapley_value_one(subject_index) for subject_index in self.subject_indices]).reshape([len(self.subject_indices),self.data.shape[1]])
    
def legal_permutations_generator(union_structure):
    #union_structure must be a list of lists of integers where each sublist represents one union
    for outer_perm in itertools.permutations(union_structure):
        for v in itertools.product(*map(lambda v: list(itertools.permutations(v)),outer_perm)):
            yield(tuple(itertools.chain(*v)))

class CohortOwen(CohortShapley):
    def __init__(self, union_structure, model, similarity, subject_indices, data, func=np.average, y=None,
            parallel=0, pid=0, data_weight=None, permutations=None,
            mc_num = None,
            verbose=1):
        super().__init__(model=model, similarity=similarity, subject_indices=subject_indices, data=data, func=func, y=y,
            parallel=parallel, pid=pid, data_weight=data_weight, permutations=permutations, mc_num=mc_num, verbose=verbose)
        self.union_structure = union_structure
        if  mc_num != None:
            n_vars = data.shape[-1]
            n_unions = len(union_structure)
            permutations = np.zeros((mc_num, n_vars), dtype=int)
            for k in range(mc_num):
                union_order = np.random.permutation(n_unions)
                perm = []
                for union_ind in union_order:
                    union = union_structure[union_ind]
                    perm = np.append(perm,np.random.permutation(union))
                permutations[k] = perm
            self.permutations = permutations
        else:
            self.permutations = np.array(list(legal_permutations_generator(union_structure)))
        self.n_permutations = self.permutations.shape[0]
