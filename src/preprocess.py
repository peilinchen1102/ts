from tokenizers.cpd import cpd_segmentation
from tokenizers.fixed import fixed_segmentation

def preprocess(time_series, fixed, window_size, overlap):

    if fixed:
        return fixed_segmentation(time_series, window_size, overlap)
    return cpd_segmentation(time_series, 'bottomup', window_size, 60, 5)
