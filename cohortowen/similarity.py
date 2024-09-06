import math
import numpy as np
import pandas as pd

# Similarity Function Example

ratio = 0.1
cutoff = 5
bins = 10

def generate_similarity_in_distance(data, subject_index):
    xmax = np.percentile(data,100-cutoff,0)
    xmin = np.percentile(data,cutoff,0)
    subject = data[subject_index]
    xdist = (xmax - xmin) * ratio
    cmin = subject - xdist
    cmax = subject + xdist
    return np.logical_and(np.greater_equal(data,cmin),np.less_equal(data,cmax))

def set_ratio(x):
    global ratio
    ratio = x

def set_cutoff(x):
    global cutoff
    cutoff - x

def set_bins(x):
    global bins
    bins = x

