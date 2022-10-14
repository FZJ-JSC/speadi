import numpy as np


def prepare_input_group(input_group):
    if not isinstance(input_group, list):
        if isinstance(input_group, np.ndarray):
            input_group = [input_group]

    # if isinstance(input_group, list):
    #     if not isinstance(input_group[0], np.ndarray):
    #         input_group = [np.array(sub_group, dtype=np.int32) for sub_group in input_group]

    input_group = [np.array(sub_group, dtype=np.int32) for sub_group in input_group]

    return input_group
