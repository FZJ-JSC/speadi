import sys
from numba import set_num_threads
set_num_threads(2)

def inspect(foo):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(f'numba_inspections/{foo.__name__}.txt', 'w') as f:
        sys.stdout = f # Redirect the standard output to a file.
        foo.inspect_types()
    sys.stdout = original_stdout


from mdenvironment.vanhove.jit.add_distribution_function_time_slice import _jit_append_Grts_self
inspect(_jit_append_Grts_self)

from mdenvironment.vanhove.jit.add_distribution_function_time_slice import _jit_append_Grts_ortho_mic
inspect(_jit_append_Grts_ortho_mic)

from mdenvironment.vanhove.jit.add_distribution_function_time_slice import _jit_append_Grts_general_mic
inspect(_jit_append_Grts_general_mic)

# from mdenvironment.time_resolved_rdf.jit.add_distribution_function_time_slice import _jit_append_grts_ortho_mic as foo
# inspect(foo)
