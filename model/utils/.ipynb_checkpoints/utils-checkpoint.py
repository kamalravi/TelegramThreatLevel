def set_seed(seed: int):
    if 'os' in sys.modules:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
    if 'random' in sys.modules:
        import random
        random.seed(seed)
    if 'numpy' in sys.modules:
        import numpy as np
        np.random.seed(seed)
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.random.set_seed(seed)