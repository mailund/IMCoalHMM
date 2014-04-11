import unittest
import IMCoalHMM.break_points


class ModuleTests(unittest.TestCase):
    def test_exp_break_points(self):
        exp_break_points = IMCoalHMM.break_points.exp_break_points

        # TODO Comments indicate the return type is a list but it isn't.
        # The return type is from expon.ppf, which is documented as "array_like".
        # self.assertIsInstance(exp_break_points(5, 1.0), list)

        # The return length should equal the first argument regardless of the rate and offset.
        for no_intervals in xrange(1, 50):
            self.assertEquals(
                len(exp_break_points(no_intervals, 1.0)),
                no_intervals)
            self.assertEquals(
                len(exp_break_points(no_intervals, 2.0, 3.0)),
                no_intervals)

        # The offset should be added to all break points.
        for no_intervals in xrange(1, 5):
            for offset in xrange(1, 10):
                for i in xrange(no_intervals):
                    self.assertAlmostEqual(
                        exp_break_points(no_intervals, 1.0)[i] + float(offset),
                        exp_break_points(no_intervals, 1.0, float(offset))[i])

        # The coalescence rate should be divided from all break points.
        for no_intervals in xrange(1, 5):
            for coal_rate in xrange(1, 10):
                for i in xrange(no_intervals):
                    self.assertAlmostEqual(
                        exp_break_points(no_intervals, float(coal_rate))[i],
                        exp_break_points(no_intervals, 1.0)[i] / coal_rate)

        # Test a few random cases.
        self.assertListEqual(
            list(exp_break_points(5, 1.0)),
            [0.0, 0.22314355131420976, 0.51082562376599072, 0.916290731874155, 1.6094379124341005])
        self.assertListEqual(
            list(exp_break_points(10, 2.0, -100.0)),
            [
                -100.0, -99.947319742171089, -99.888428224342888, -99.821662528030629,
                -99.744587188117009, -99.653426409720026, -99.541854634062929,
                -99.398013597837036, -99.195281043782956, -98.848707453502982
            ])

    def test_uniform_break_points(self):
        uniform_break_points = IMCoalHMM.break_points.uniform_break_points

        # TODO Comments indicate the return type is a list but it isn't.
        # The return type is from uniform.ppf, which is documented as "array_like".
        # self.assertIsInstance(uniform_break_points(5, 1.0, 3.0), list)

        # The return length should equal the first argument regardless of the start and end.
        for no_intervals in xrange(1, 5):
            self.assertEquals(
                len(uniform_break_points(no_intervals, 1.0, 2.0)),
                no_intervals)
            self.assertEquals(
                len(uniform_break_points(no_intervals, 100.0, 200.0)),
                no_intervals)

        # The start should be the first break point.
        for no_intervals in xrange(1, 5):
            for start in xrange(10):
                for end in xrange(start + 1, 10):
                    self.assertAlmostEqual(
                        uniform_break_points(no_intervals, float(start), float(end))[0],
                        float(start))

        # The start should be less than or equal to all break points.
        for no_intervals in xrange(1, 5):
            for start in xrange(10):
                for end in xrange(start + 1, 10):
                    for i in xrange(no_intervals):
                        self.assertGreaterEqual(
                            uniform_break_points(no_intervals, float(start), float(end))[i],
                            float(start))

        # The end should be greater than all break points.
        for no_intervals in xrange(1, 5):
            for start in xrange(10):
                for end in xrange(start + 1, 10):
                    for i in xrange(no_intervals):
                        self.assertGreater(
                            float(end),
                            uniform_break_points(no_intervals, float(start), float(end))[i])

        # Test a few random cases.
        self.assertListEqual(
            list(uniform_break_points(7, 1.0, 50.0)),
            [1.0, 8.0, 15.0, 22.0, 29.0, 36.0, 43.0])
        self.assertListEqual(
            list(uniform_break_points(10, -20.0, 100.0)),
            [-20.0, -8.0, 4.0, 16.0, 28.0, 40.0, 52.0, 64.0, 76.0, 88.0])

    def test_psmc_break_points(self):
        psmc_break_points = IMCoalHMM.break_points.psmc_break_points

        # The return type should be a list.
        self.assertIsInstance(psmc_break_points(), list)

        # The return length should equal the first argument regardless of the other params.
        for no_intervals in xrange(1, 5):
            self.assertEquals(
                len(psmc_break_points(no_intervals)),
                no_intervals)
            self.assertEquals(
                len(psmc_break_points(no_intervals, 100.0, 0.1, 200.0)),
                no_intervals)

        # The offset should be added to all break points.
        for no_intervals in xrange(1, 5):
            for offset in xrange(1, 10):
                for i in xrange(no_intervals):
                    self.assertAlmostEqual(
                        psmc_break_points(no_intervals)[i] + float(offset),
                        psmc_break_points(no_intervals, offset=float(offset))[i])

        # The offset should be the first break point.
        for no_intervals in xrange(1, 5):
            for t_max in xrange(50, 5):
                for mu_m in xrange(10):
                    for offset in xrange(100, 20):
                        self.assertAlmostEqual(
                            psmc_break_points(no_intervals, float(t_max), mu_m / 100000.0, float(offset))[0],
                            float(offset))

        # The offset should be less than or equal to all break points.
        for no_intervals in xrange(1, 5):
            for t_max in xrange(0, 50, 5):
                for mu_m in xrange(10):
                    for offset in xrange(0, 100, 20):
                        for i in xrange(no_intervals):
                            self.assertGreaterEqual(
                                psmc_break_points(no_intervals, float(t_max), mu_m / 100000.0, float(offset))[i],
                                float(offset))

        # Test a range of arguments returns expected values.
        for no_intervals in xrange(1, 5):
            for t_max in xrange(0, 50, 5):
                for mu_m in xrange(10):
                    for offset in xrange(0, 100, 20):
                        for i in xrange(1, no_intervals):
                            mu = mu_m / 100000.0
                            self.assertAlmostEqual(
                                psmc_break_points(no_intervals, float(t_max), mu, float(offset))[i],
                                offset + 0.1 * ((1.0 + 10.0 * t_max * mu)**(float(i) / no_intervals) - 1.0))

        # Test a few random cases.
        self.assertListEqual(
            list(psmc_break_points(4)),
            [0.0, 3.7499997995738e-09, 7.499999710169902e-09, 1.124999979840169e-08])
        self.assertListEqual(
            list(psmc_break_points(4, 5, 1)),
            [0.0, 0.16723451177837886, 0.614142842854285, 1.8084361395018835])
        self.assertListEqual(
            list(psmc_break_points(4, 50, 0.1)),
            [0.0, 0.16723451177837886, 0.614142842854285, 1.8084361395018835])
