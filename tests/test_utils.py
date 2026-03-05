import utils
import numpy as np


def test_get_true_distribution_retrocompatibility():
    num_nodes, _ = utils.get_total_number("./data/ICEWS14", "stat.txt")
    train_data, _ = utils.load_quadruples("./data/ICEWS14", "train.txt")
    new = utils.get_true_distribution(train_data, num_nodes)
    old = utils.get_true_distribution_old(train_data, num_nodes)
    # NOTE: the old implementation had a bug where computation for the
    # last timestep is incorrect, hence the difference. Also, atol is
    # relatively high since the original implementation falsely count
    # the first timeline of the next timestamp for the previous
    # timestamp.
    assert np.allclose(new[0][:-1], old[0][:-1], rtol=0, atol=1e-1)
