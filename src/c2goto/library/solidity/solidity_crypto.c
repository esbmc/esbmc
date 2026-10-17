/* [APPROX: OVER + UNDER] Solidity crypto hashes — deterministic
 * bijective identity abstraction. Each hash is a simple bitwise xor:
 *   keccak256(x) = ~x, sha256(x) = ~(x+1), ripemd160(x) = ~(x+2),
 *   ecrecover(h,v,r,s) = ~h.
 *
 * Modeling properties:
 *  - Functional consistency (same input → same output): keccak256(x) ==
 *    keccak256(x) is provable.
 *  - Injectivity (different inputs → different outputs): keccak256(a) ==
 *    keccak256(b) iff a == b. This is OVER-approximate relative to a real
 *    hash (no collisions exist here that a real hash would expose, but the
 *    frontend never relies on finding a collision so it is sound).
 *  - Distinct hash families: keccak256(x) != sha256(x) for x != 0.
 *  - Concrete hash values: NOT computed. Any property of the form
 *    `keccak256(0) == 0xc5d2...` is UNPROVABLE — UNDER-approximate for
 *    reasoning that depends on the real bit pattern of a specific hash.
 *  - ecrecover: returns a deterministic function of `hash` only, ignoring
 *    (v,r,s). An attacker model that requires signature forgery is not
 *    captured — UNDER-approximate for auth-bypass properties.
 *
 * Use cases covered:
 *  ✓ Equality-based reasoning (set membership via hashed keys)
 *  ✓ Uniqueness of derived IDs
 * Use cases NOT covered:
 *  ✗ Specific hash bit patterns (preimage witness, checksum matching)
 *  ✗ Signature-verification semantics of ecrecover
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

/* F1 closure (ledger #3): wide-BV-indexed hash tables.
 *
 * The Solidity frontend concatenates abi.encode/keccak256/sha256 args into
 * a wide BV `concat(arg_1, ..., arg_N)`, picks the smallest enclosing bucket
 * W ∈ {256, 512, 1024, 2048}, zero-extends to W, and looks up
 * `_ESBMC_<hash>_table_<W>[concat_W]`.  The SMT array axiom gives the
 * `same key → same hash` direction for free; per-callsite distinctness
 * assumes emitted at the frontend cover the `distinct keys → distinct
 * hashes` direction (injectivity).
 *
 * The 4-bucket schedule covers the common signature widths:
 *   - 1 × uint256        →  256
 *   - 2 × uint256        →  512
 *   - 3-4 × uint256      → 1024
 *   - 5-8 × uint256      → 2048
 * Heterogeneous mixes (e.g. (address, uint256, uint256) = 576 bits) round
 * UP to the smallest enclosing bucket and zero-extend in the unused high
 * bits.  Distinct concat values still map to distinct slots because the
 * zero-extension is a bijection on the lower `total_W` bits.
 *
 * Replaces the unsound multiplicative `_ESBMC_abi_fold` (deleted in S5).
 */
__attribute__((
  annotate("__ESBMC_inf_size:256"))) uint256_t _ESBMC_keccak_table_256[1];
__attribute__((
  annotate("__ESBMC_inf_size:512"))) uint256_t _ESBMC_keccak_table_512[1];
__attribute__((
  annotate("__ESBMC_inf_size:1024"))) uint256_t _ESBMC_keccak_table_1024[1];
__attribute__((
  annotate("__ESBMC_inf_size:2048"))) uint256_t _ESBMC_keccak_table_2048[1];

__attribute__((
  annotate("__ESBMC_inf_size:256"))) uint256_t _ESBMC_sha256_table_256[1];
__attribute__((
  annotate("__ESBMC_inf_size:512"))) uint256_t _ESBMC_sha256_table_512[1];
__attribute__((
  annotate("__ESBMC_inf_size:1024"))) uint256_t _ESBMC_sha256_table_1024[1];
__attribute__((
  annotate("__ESBMC_inf_size:2048"))) uint256_t _ESBMC_sha256_table_2048[1];

__attribute__((
  annotate("__ESBMC_inf_size:256"))) address_t _ESBMC_ripemd160_table_256[1];
__attribute__((
  annotate("__ESBMC_inf_size:512"))) address_t _ESBMC_ripemd160_table_512[1];
__attribute__((
  annotate("__ESBMC_inf_size:1024"))) address_t _ESBMC_ripemd160_table_1024[1];
__attribute__((
  annotate("__ESBMC_inf_size:2048"))) address_t _ESBMC_ripemd160_table_2048[1];
