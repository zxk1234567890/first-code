import numpy as np
import  torch
from torch.utils import data
from Linear import synthetic_data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
