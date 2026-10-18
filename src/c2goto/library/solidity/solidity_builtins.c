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

/* integer power: base**exp using binary exponentiation */
uint256_t sol_pow_uint(uint256_t base, uint256_t exp)
{
__ESBMC_HIDE:;
  uint256_t result = 1;
  while (exp > 0)
  {
    if (exp & 1)
      result *= base;
    base *= base;
    exp >>= 1;
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

/*
 * llc_nondet_bytes — nondet abstraction for a `bytes memory` value whose
 * content is unknown but whose structural invariants are known to hold.
 * Used for (a) the data component of a low-level .call()/.staticcall()/
 * .delegatecall() return, and (b) entry-harness parameters of type
 * `bytes calldata` / `bytes memory` (via assign_param_nondet).
 *
 * The returned BytesDynamic is constrained so:
 *  - `initialized == 1` — init-checks pass unconditionally
 *  - `length` is fully nondet (size_t range, scalar-driven by the SMT
 *    solver per assertion / counterexample). Real Solidity admits any
 *    gas-bounded length, including 0; contracts that need a tighter
 *    range must require() it explicitly.
 *  - `capacity == length` — matches a freshly-decoded calldata view
 *  - `offset` is fresh nondet — pool contents remain symbolic.
 */
BytesDynamic llc_nondet_bytes(void)
{
__ESBMC_HIDE:;
  BytesDynamic result;
  result.initialized = 1;
  result.capacity = result.length;
  return result;
}

/* selfdestruct */
void selfdestruct()
{
__ESBMC_HIDE:;
  exit(0);
}
