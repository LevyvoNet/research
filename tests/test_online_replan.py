import unittest

from solvers.online_replan import divide_to_groups


class IdTests(unittest.TestCase):
    def test_divide_to_groups(self):
        locs = ((0, 0), (0, 2), (0, 4), (0, 7))

        groups = divide_to_groups(locs, 2)

        self.assertListEqual(groups, [[0, 1, 2], [3]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
