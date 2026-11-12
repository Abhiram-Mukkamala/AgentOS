import unittest
from math_utils import factorial, is_prime, list_primes


class TestMathUtils(unittest.TestCase):
    # ----- factorial tests -------------------------------------------------
    def test_factorial_basic(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)
        self.assertEqual(factorial(5), 120)

    def test_factorial_negative(self):
        with self.assertRaises(ValueError):
            factorial(-3)

    # ----- is_prime tests -------------------------------------------------
    def test_is_prime_small_numbers(self):
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertFalse(is_prime(4))
        self.assertTrue(is_prime(13))
        self.assertFalse(is_prime(15))

    def test_is_prime_large_number(self):
        self.assertTrue(is_prime(7919))
        self.assertFalse(is_prime(7920))

    # ----- list_primes tests -----------------------------------------------
    def test_list_primes_up_to_10(self):
        self.assertEqual(list_primes(10), [2, 3, 5, 7])

    def test_list_primes_up_to_2(self):
        self.assertEqual(list_primes(2), [2])

    def test_list_primes_zero(self):
        self.assertEqual(list_primes(0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
