"""Statistical measurements and tests.
"""

import statys.tests.measure as m
import statys.tests.wilcoxon as w
from statys.core import Distribution


def measurement_report(*args):
    """Performs statistical measurements (mean and standard deviation)
    and returns a report containing the results.

    Note that one can input an unlimited amount of list-based arguments.

    Returns:
        Dictionaries holding mean and standard deviation of input arguments.

    """

    # Instantiates a Distribution-based class
    d = Distribution(*args)

    # Calculates the mean and standard deviation of the distribution
    mean = m.mean(d)
    std = m.std(d)

    return mean, std


def wilcoxon_report(*args):
    """Performs the Wilcoxon-based tests and returns a report containing the results.

    Note that one can input an unlimited amount of list-based arguments.

    Returns:
        Dictionaries holding h-index and p-value of tests over the input arguments.

    """

    # Instantiates a Distribution-based class
    d = Distribution(*args)

    # Performs the Wilcoxon signed-rank and rank-sum tests
    signed_rank = w.signed_rank(d)
    rank_sum = w.rank_sum(d)

    return signed_rank, rank_sum
