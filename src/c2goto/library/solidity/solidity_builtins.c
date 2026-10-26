/*
 * Solidity built-in utility functions.
 *
 * Contains: integer exponentiation (sol_pow_uint), modular arithmetic
 * (addmod/mulmod with 512-bit arbitrary-precision intermediates per spec),
 * low-level call nondet bytes abstraction, and selfdestruct.
 *
 * Block/transaction/message context variables and functions are in
 * solidity_blockchain.c.  Cryptographic hash functions are in
 * solidity_crypto.c.
 */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "solidity_types.h"

/* integer power: base**exp using binary exponentiation. base is not squared
 * after the last exponent bit, so a multiplication overflows only when the
 * result does. */
uint256_t sol_pow_uint(uint256_t base, uint256_t exp)
{
__ESBMC_HIDE:;
  uint256_t result = 1;
  while (exp > 0)
  {
    if (exp & 1)
      result *= base;
    exp >>= 1;
    if (exp > 0)
      base *= base;
  }
  return result;
}

int256_t sol_pow_int(int256_t base, uint256_t exp)
{
__ESBMC_HIDE:;
  int256_t result = 1;
  while (exp > 0)
  {
    if (exp & 1)
      result *= base;
    exp >>= 1;
    if (exp > 0)
      base *= base;
  }
  return result;
}

/*
 * Modular arithmetic — arbitrary precision per Solidity spec.
 *
 * Solidity specifies that addmod/mulmod perform the intermediate
 * addition/multiplication with arbitrary precision (no wrap at 2^256).
 * We use a 512-bit intermediate type to avoid overflow.
 */
typedef unsigned BIGINT(512) uint512_t;

uint256_t addmod(uint256_t x, uint256_t y, uint256_t k)
{
__ESBMC_HIDE:;
  uint512_t wide = (uint512_t)x + (uint512_t)y;
  return (uint256_t)(wide % (uint512_t)k);
}

uint256_t mulmod(uint256_t x, uint256_t y, uint256_t k)
{
__ESBMC_HIDE:;
  uint512_t wide = (uint512_t)x * (uint512_t)y;
  return (uint256_t)(wide % (uint512_t)k);
}

/* base**exp modulo 2^256, for an unchecked block: no 512-bit product of two
 * 256-bit values overflows. Signed operands wrap to the same bits. */
uint256_t sol_pow_wrap(uint256_t base, uint256_t exp)
{
__ESBMC_HIDE:;
  uint256_t result = 1;
  while (exp > 0)
  {
    if (exp & 1)
      result = (uint256_t)((uint512_t)result * base);
    exp >>= 1;
    if (exp > 0)
      base = (uint256_t)((uint512_t)base * base);
  }
  return result;
}

/*
 * llc_nondet_bytes — nondet abstraction for the `bytes memory data` component
 * of a low-level `.call()` / `.staticcall()` / `.delegatecall()` return.
 *
 * The returned BytesDynamic has `initialized = 1` so that init-checks pass.
 * All other fields (length, offset, capacity) are left uninitialized →
 * ESBMC treats them as fresh nondet symbols.  In particular, `data.length`
 * is fully unconstrained (any size_t value is possible).
 *
 * Over-approximation semantics (same as crypto hash abstraction):
 *  - Sound for safety: no real return-data outcome is excluded.
 *  - Possible false positives for properties that rely on the concrete
 *    content or on specific length values.
 */
BytesDynamic llc_nondet_bytes(void)
{
__ESBMC_HIDE:;
  BytesDynamic result;
  result.initialized = 1;
  return result;
}

/* selfdestruct */
void selfdestruct()
{
__ESBMC_HIDE:;
  exit(0);
}
