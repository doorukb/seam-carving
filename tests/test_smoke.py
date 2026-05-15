import numpy as np

import seam_carver as sc


def test_import_and_vertical_seam_runs():
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    s = sc.find_vertical_seam(img)
    assert s.shape == (4,)
