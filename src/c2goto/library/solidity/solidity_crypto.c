/* Solidity cryptographic hash functions — deterministic bijective abstraction.
 *
 * Each function is modeled as a simple deterministic transformation of its
 * input.  This provides:
 *  - Functional consistency: same input always yields the same output,
 *    so keccak256(x) == keccak256(x) is provable.
 *  - Injectivity: different inputs yield different outputs (bijective),
 *    so keccak256(a) == keccak256(b) iff a == b.
 *  - O(1) for the SMT solver: single bitvector operation, no arrays or loops.
 *
 * Trade-offs:
 *  - The concrete hash value is not computed; assertions about specific
 *    hash outputs (e.g. keccak256(0) == 0xc5d2...) will not be provable.
 *  - The abstraction is sound for equality-based reasoning (e.g. string
 *    comparison via keccak256(abi.encodePacked(s1)) == keccak256(...s2)).
 *
 * Each hash function uses a distinct transformation to ensure
 * keccak256(x) != sha256(x) for all x != 0.
 */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "solidity_types.h"

uint256_t keccak256(uint256_t x)
{
__ESBMC_HIDE:;
  return ~x;
}

uint256_t sha256(uint256_t x)
{
__ESBMC_HIDE:;
  return ~(x + 1);
}

address_t ripemd160(uint256_t x)
{
__ESBMC_HIDE:;
  return (address_t)(~(x + 2));
}

address_t ecrecover(uint256_t hash, unsigned int v, uint256_t r, uint256_t s)
{
__ESBMC_HIDE:;
  return (address_t)(~hash);
}
