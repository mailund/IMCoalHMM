import unittest
import IMCoalHMM.statespace_generator

#
# A "state" is a frozenset of "tokens".
#
# A "token" is a tuple with:
#     [0] an integer representing the population
#     [1] a tuple with:
#         [0] a frozenset of integers representing left nucleotides
#         [1] a frozenset of integers representing right nucleotides
#
# CoalSystem.transitions is a tuple with:
#     [0] a list of tuples with:
#         [0] a letter identifier for the transition type
#         [1] a function that returns an array of tuples* for a given state
#     [1] a list of tuples with:
#         [0] a letter identifier for the transition type
#         [1] a function that returns one tuple* for a given state
#
# * The values returned from the transition functions are tuples with:
#     [0] A letter identifier for the transition type
#     [1] An integer representing the source population
#     [2] An integer representing the destination population
#     [3] A state in the format described above.
#


def _freeze_token(token):
    population, (left_nucleotides, right_nucleotides) = token
    return population, (frozenset(left_nucleotides), frozenset(right_nucleotides))


def _freeze_state(state):
    return frozenset([_freeze_token(token) for token in state])


class ModuleTests(unittest.TestCase):
    def test_has_left_coalesced(self):
        has_left_coalesced = IMCoalHMM.statespace_generator.has_left_coalesced

        # The order does not affect the result; try index 0.
        self.assertTrue(has_left_coalesced(_freeze_state([
            (1, ([1, 2], [])),
            (1, ([], [3])),
            (1, ([], [4]))
        ])))

        # The order does not affect the result; try index 1.
        self.assertTrue(has_left_coalesced(_freeze_state([
            (1, ([], [3])),
            (1, ([1, 2], [])),
            (1, ([], [4]))
        ])))

        # The order does not affect the result; try index 2.
        self.assertTrue(has_left_coalesced(_freeze_state([
            (1, ([], [3])),
            (1, ([], [4])),
            (1, ([1, 2], []))
        ])))

        # Both sides may coalesce.
        self.assertTrue(has_left_coalesced(_freeze_state([
            (1, ([1, 2], [3, 4]))
        ])))

        # Even if right has coalesced, left must or else it returns False.
        self.assertFalse(has_left_coalesced(_freeze_state([
            (1, ([1], [])),
            (1, ([2], [])),
            (1, ([], [3, 4]))
        ])))

    def test_has_right_coalesced(self):
        has_right_coalesced = IMCoalHMM.statespace_generator.has_right_coalesced

        # The order does not affect the result; try index 0.
        self.assertTrue(has_right_coalesced(_freeze_state([
            (1, ([], [1, 2])),
            (1, ([3], [])),
            (1, ([4], []))
        ])))

        # The order does not affect the result; try index 1.
        self.assertTrue(has_right_coalesced(_freeze_state([
            (1, ([3], [])),
            (1, ([], [1, 2])),
            (1, ([4], []))
        ])))

        # The order does not affect the result; try index 2.
        self.assertTrue(has_right_coalesced(_freeze_state([
            (1, ([3], [])),
            (1, ([4], [])),
            (1, ([], [1, 2]))
        ])))

        # Both sides may coalesce.
        self.assertTrue(has_right_coalesced(_freeze_state([
            (1, ([3, 4], [1, 2]))
        ])))

        # Even if left has coalesced, right must or else it returns False.
        self.assertFalse(has_right_coalesced(_freeze_state([
            (1, ([], [1])),
            (1, ([], [2])),
            (1, ([3, 4], []))
        ])))


class CoalSystemTests(unittest.TestCase):
    def test_init(self):
        system = IMCoalHMM.statespace_generator.CoalSystem()
        self.assertIsNone(system.state_numbers)
        self.assertIsNone(system.states)
        self.assertIsNone(system.init)
        self.assertListEqual([], system.transitions)
        self.assertListEqual([], system.begin_states)
        self.assertListEqual([], system.left_states)
        self.assertListEqual([], system.right_states)
        self.assertListEqual([], system.end_states)

    def test_successors(self):
        coalesce = IMCoalHMM.statespace_generator.CoalSystem.coalesce
        recombination = IMCoalHMM.statespace_generator.CoalSystem.recombination

        system = IMCoalHMM.statespace_generator.CoalSystem()

        # Test that without any transitions defined, no new states return.
        s = _freeze_state([(9, ([1], [])), (9, ([2], [])), (9, ([], [3])), (9, ([], [4]))])
        system.transitions = ((), ())
        self.assertListEqual(list(system.successors(s)), [])

        # Test recombination for a simple state.
        s = _freeze_state([(9, ([1], [3])), (9, ([2], [4]))])
        system.transitions = ((('R', recombination),), ())
        self.assertListEqual(list(system.successors(s)), [
            ('R', 9, 9, _freeze_state([(9, ([1], [])), (9, ([2], [4])), (9, ([], [3]))])),
            ('R', 9, 9, _freeze_state([(9, ([1], [3])), (9, ([2], [])), (9, ([], [4]))]))
        ])

        # Test coalescence for a simple state.
        s = _freeze_state([(9, ([1], [])), (9, ([], [2]))])
        system.transitions = ((), (('C', coalesce),))
        self.assertListEqual(list(system.successors(s)), [
            ('C', 9, 9, _freeze_state([(9, ([1], [2]))]))
        ])

        # Test both types of transitions together.
        t8 = (8, ([5, 6], []))
        s = _freeze_state([(9, ([1], [3])), (9, ([2], [4])), t8])
        system.transitions = ((('R', recombination),), (('C', coalesce),))
        self.assertListEqual(list(system.successors(s)), [
            ('R', 9, 9, _freeze_state([(9, ([1], [])), (9, ([], [3])), (9, ([2], [4])), t8])),
            ('R', 9, 9, _freeze_state([(9, ([1], [3])), (9, ([2], [])), (9, ([], [4])), t8])),
            ('C', 9, 9, _freeze_state([(9, ([1, 2], [3, 4])), t8]))
        ])

    def test_compute_state_space(self):
        # TODO Test compute_state_space.
        pass

    def test_recombination(self):
        recombination = IMCoalHMM.statespace_generator.CoalSystem.recombination

        def freeze_output(tokens):
            population = tokens[0][0]
            state = frozenset([(_freeze_token(token)) for token in tokens])
            return [(population, population, state)]

        # If either side is empty, the function returns [].
        self.assertListEqual([], recombination(_freeze_token((9, ([1], [])))))
        self.assertListEqual([], recombination(_freeze_token((9, ([], [1])))))
        self.assertListEqual([], recombination(_freeze_token((9, ([1, 2], [])))))
        self.assertListEqual([], recombination(_freeze_token((9, ([], [1, 2])))))

        # Test a pair of nucleotides.
        self.assertListEqual(
            recombination(_freeze_token((9, ([1], [2])))),
            freeze_output([
                (9, ([1], [])),
                (9, ([], [2]))
            ]))

        # Test the behavior for a pair of nucleotides, both coalesced vertically.
        self.assertListEqual(
            recombination(_freeze_token((9, ([1, 2], [3, 4])))),
            freeze_output([
                (9, ([], [3, 4])),
                (9, ([1, 2], []))
            ]))

    def test_coalesce(self):
        coalesce = IMCoalHMM.statespace_generator.CoalSystem.coalesce

        t0 = _freeze_token((0, ([1], [])))
        t1 = _freeze_token((9, ([1], [])))
        t2 = _freeze_token((9, ([2], [])))
        t3 = _freeze_token((9, ([], [3])))
        t4 = _freeze_token((9, ([], [4])))

        # The function returns (-1, -1, None) when the populations differ.
        self.assertEqual(coalesce(t0, t1), (-1, -1, None))

        def freeze_output(token):
            population = token[0]
            return population, population, _freeze_state([token])

        # Test coalescing from left to right.
        self.assertEqual(coalesce(t1, t3), freeze_output((9, ([1], [3]))))
        self.assertEqual(coalesce(t1, t4), freeze_output((9, ([1], [4]))))
        self.assertEqual(coalesce(t2, t3), freeze_output((9, ([2], [3]))))
        self.assertEqual(coalesce(t2, t4), freeze_output((9, ([2], [4]))))

        # Test coalescing from right to left.
        self.assertEqual(coalesce(t3, t1), freeze_output((9, ([1], [3]))))
        self.assertEqual(coalesce(t4, t1), freeze_output((9, ([1], [4]))))
        self.assertEqual(coalesce(t3, t2), freeze_output((9, ([2], [3]))))
        self.assertEqual(coalesce(t4, t2), freeze_output((9, ([2], [4]))))

        # Test coalescing from top to bottom.
        self.assertEqual(coalesce(t1, t2), freeze_output((9, ([1, 2], []))))
        self.assertEqual(coalesce(t3, t4), freeze_output((9, ([], [3, 4]))))

        # Test coalescing from bottom to top.
        self.assertEqual(coalesce(t2, t1), freeze_output((9, ([1, 2], []))))
        self.assertEqual(coalesce(t4, t3), freeze_output((9, ([], [3, 4]))))
