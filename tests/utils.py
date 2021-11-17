from numpy.testing import assert_allclose


def compare_arrays(a, b, atol=1e-2)
    try:
        np.testing.assert_allclose(a, b, atol=1e-2)
        print(f'All elements match within {atol=}!')
    except AssertionError as err:
        print(err)
