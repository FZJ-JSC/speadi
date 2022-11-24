import sys
from numba import set_num_threads
set_num_threads(2)

def inspect(foo):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(f'numba_inspections/{foo.__name__}.txt', 'w') as f:
        sys.stdout = f # Redirect the standard output to a file.
        foo.inspect_types()
    sys.stdout = original_stdout


from speadi.vanhove.numba.add_distribution_function_time_slice import _append_Grts_self
inspect(_append_Grts_self)

from speadi.vanhove.numba.add_distribution_function_time_slice import _append_Grts_ortho_mic
inspect(_append_Grts_ortho_mic)

from speadi.vanhove.numba.add_distribution_function_time_slice import _append_Grts_general_mic
inspect(_append_Grts_general_mic)

# from speadi.time_resolved_rdf.numba.add_distribution_function_time_slice import _append_grts_mic as foo
# inspect(foo)
