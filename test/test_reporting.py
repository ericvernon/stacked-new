from pathlib import Path

import unittest

from src.reporting import parse_results_dict


class MyTestCase(unittest.TestCase):
    # Test the basic functionality using a short (random) file simulating a binary grader, with hand-counted results
    def test_results_binary_grader(self):
        test_file = Path('./data/simple_binary.txt')
        results = parse_results_dict(test_file)

        self.assertIn('hybrid_n_correct_total', results)
        self.assertIn('hybrid_n_correct_easy', results)
        self.assertIn('hybrid_n_correct_hard', results)
        self.assertIn('glass_n_correct_total', results)
        self.assertIn('glass_n_correct_easy', results)
        self.assertIn('glass_n_correct_hard', results)
        self.assertIn('black_n_correct_total', results)
        self.assertIn('black_n_correct_easy', results)
        self.assertIn('black_n_correct_hard', results)
        self.assertIn('n_total', results)
        self.assertIn('n_easy', results)
        self.assertIn('n_hard', results)

        self.assertEqual(results['n_total'], 20)
        self.assertEqual(results['n_easy'], 16)
        self.assertEqual(results['n_hard'], 4)

        self.assertEqual(results['glass_n_correct_total'], 12)
        self.assertEqual(results['glass_n_correct_easy'], 9)
        self.assertEqual(results['glass_n_correct_hard'], 3)

        self.assertEqual(results['black_n_correct_total'], 19)
        self.assertEqual(results['black_n_correct_easy'], 15)
        self.assertEqual(results['black_n_correct_hard'], 4)

        self.assertEqual(results['hybrid_n_correct_total'], 13)
        self.assertEqual(results['hybrid_n_correct_easy'], 9)
        self.assertEqual(results['hybrid_n_correct_hard'], 4)

    def test_results_ternary_grader(self):
        # A random data file simulating the ternary grader. Expected results are hand counted.
        test_file = Path('./data/simple_ternary.txt')
        results = parse_results_dict(test_file)

        self.assertIn('hybrid_n_correct_total', results)
        self.assertIn('hybrid_n_correct_easy', results)
        self.assertIn('hybrid_n_correct_hard', results)
        self.assertIn('hybrid_n_reject', results)
        self.assertIn('glass_n_correct_total', results)
        self.assertIn('glass_n_correct_easy', results)
        self.assertIn('glass_n_correct_hard', results)
        self.assertIn('glass_n_correct_very_hard', results)
        self.assertIn('black_n_correct_total', results)
        self.assertIn('black_n_correct_easy', results)
        self.assertIn('black_n_correct_hard', results)
        self.assertIn('black_n_correct_very_hard', results)
        self.assertIn('n_total', results)
        self.assertIn('n_easy', results)
        self.assertIn('n_hard', results)
        self.assertIn('n_very_hard', results)

        self.assertEqual(results['n_total'], 20)
        self.assertEqual(results['n_easy'], 10)
        self.assertEqual(results['n_hard'], 6)
        self.assertEqual(results['n_very_hard'], 4)

        self.assertEqual(results['glass_n_correct_total'], 16)
        self.assertEqual(results['glass_n_correct_easy'], 8)
        self.assertEqual(results['glass_n_correct_hard'], 6)
        self.assertEqual(results['glass_n_correct_very_hard'], 2)

        self.assertEqual(results['black_n_correct_total'], 18)
        self.assertEqual(results['black_n_correct_easy'], 9)
        self.assertEqual(results['black_n_correct_hard'], 5)
        self.assertEqual(results['black_n_correct_very_hard'], 4)

        self.assertEqual(results['hybrid_n_correct_total'], 13)
        self.assertEqual(results['hybrid_n_correct_easy'], 8)
        self.assertEqual(results['hybrid_n_correct_hard'], 5)
        self.assertEqual(results['hybrid_n_reject'], 4)

    # Test basic assumptions about the data structure using a long (random generated) file
    def test_results_long(self):
        test_file = Path('./data/simple_ternary_long.txt')
        results = parse_results_dict(test_file)

        self.assertEqual(results['n_total'], results['n_easy'] + results['n_hard'] + results['n_very_hard'])
        self.assertEqual(results['n_very_hard'], results['hybrid_n_reject'])

        self.assertEqual(results['glass_n_correct_total'],
                         results['glass_n_correct_easy'] + results['glass_n_correct_hard'] + results['glass_n_correct_very_hard'])
        self.assertEqual(results['black_n_correct_total'],
                         results['black_n_correct_easy'] + results['black_n_correct_hard'] + results['black_n_correct_very_hard'])
        self.assertEqual(results['hybrid_n_correct_total'],
                         results['hybrid_n_correct_easy'] + results['hybrid_n_correct_hard'])

        self.assertEqual(results['glass_n_correct_easy'], results['hybrid_n_correct_easy'])
        self.assertEqual(results['black_n_correct_hard'], results['hybrid_n_correct_hard'])


if __name__ == '__main__':
    unittest.main()
