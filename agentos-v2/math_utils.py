"""Utility functions for basic mathematics.

Provides:
- `factorial(n)`: Returns n! for non‑negative integers.
- `is_prime(n)`: Returns True if n is a prime number.
- `list_primes(limit)`: Returns a list of all prime numbers up to `limit` (inclusive).
"""

from math import prod
from typing import List


def factorial(n: int) -> int:
    """Return the factorial of a non‑negative integer `n`.

    Raises:
        ValueError: If `n` is negative.
    """
    if n < 0:
        raise ValueError("factorial is not defined for negative numbers")
    if n in (0, 1):
        return 1
    # Use `prod` for a concise product of the range 1..n
    return prod(range(1, n + 1))


def is_prime(n: int) -> bool:
    """Return True if `n` is a prime number, False otherwise.

    Handles edge cases for numbers < 2.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Check odd divisors up to sqrt(n)
    limit = int(n ** 0.5) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def list_primes(limit: int) -> List[int]:
    """Return a list of all prime numbers up to `limit` (inclusive)."""
    return [i for i in range(2, limit + 1) if is_prime(i)]
