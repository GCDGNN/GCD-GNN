import numpy as np
import torch as tc
import random
import os
import dgl

def set_random_seed(seed):	
	os.environ['PYTHONHASHSEED'] = str(seed)	
	random.seed(seed)
	np.random.seed(seed)
	tc.manual_seed(seed)
	tc.cuda.manual_seed(seed)
	tc.cuda.manual_seed_all(seed)
	tc.backends.cudnn.deterministic = True
	tc.backends.cudnn.benchmark = False
	# torch.manual_seed(seed)
	# np.random.seed(seed)
	# random.seed(seed)
	# os.environ['PYTHONHASHSEED'] = str(seed)
	dgl.seed(seed)
	dgl.random.seed(seed)
	# tc.use_deterministic_algorithms(True)
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	tc.set_num_threads(1)