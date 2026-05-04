/*
 * Solidity ABI Encoding and Decoding — operational models.
 *
 * ABI encoding functions use an **identity abstraction**: they return the
 * first argument unchanged.  This is sound for equality-based reasoning
 * (e.g. keccak256(abi.encodePacked(x)) == keccak256(abi.encodePacked(x)))
 * because the transformation is deterministic and injective.
 *
 * Multi-argument calls: only the first argument is captured by the model;
 * remaining arguments are evaluated for side effects but discarded.
 * This is an over-approximation — properties that depend on the packed
 * byte layout of multiple arguments may yield false positives.
 *
 * abi.decode is modeled as nondet (over-approximation): the decoded values
 * are unconstrained.  This is sound because any concrete decoded value is
 * a possible result of the nondet abstraction.
 */

#include <stdint.h>
#include "solidity_types.h"

/* ── abi.encode(...)  ────────────────────────────────────────────────── */
uint256_t abi_encode(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}

/* ── abi.encodePacked(...)  ──────────────────────────────────────────── */
uint256_t abi_encodePacked(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}

/* ── abi.encodeWithSelector(bytes4 selector, ...)  ───────────────────── */
uint256_t abi_encodeWithSelector(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}

/* ── abi.encodeWithSignature(string memory signature, ...)  ──────────── */
uint256_t abi_encodeWithSignature(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}

/* ── abi.encodeCall(function, (...))  ────────────────────────────────── */
uint256_t abi_encodeCall(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}

/* ── abi.decode(bytes memory, (types))  ──────────────────────────────── *
 *
 * In real Solidity, abi.decode unpacks ABI-encoded bytes into typed values.
 * We model it as returning a nondet uint256 — an over-approximation that
 * is sound for safety properties.  The caller cannot assume any specific
 * decoded value, so no real bug is masked.
 *
 * When the frontend encounters abi.decode with a tuple return, each
 * component is independently nondet.
 */
uint256_t abi_decode(uint256_t x)
{
__ESBMC_HIDE:;
  uint256_t result;
  return result;
}
