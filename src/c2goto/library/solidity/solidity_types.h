#ifndef SOLIDITY_TYPES_H
#define SOLIDITY_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__clang__)
#  if __clang_major__ >= 16
#    define BIGINT(bits) _BitInt(bits)
#  elif __clang_major__ >= 11 && __clang_major__ <= 13
#    define BIGINT(bits) _ExtInt(bits)
#  else
#    error "Unsupported Clang version: _ExtInt/_BitInt not available."
#  endif
#else
#  error "This code requires Clang to compile."
#endif

typedef BIGINT(256) int256_t;
typedef unsigned BIGINT(256) uint256_t;
typedef unsigned BIGINT(160) address_t;

/* Dynamic bytes type — shared between solidity_bytes.c and solidity_builtins.c
 */
typedef struct BytesDynamic
{
  size_t offset;
  size_t length;
  size_t capacity;
  int initialized;
} BytesDynamic;

struct sol_llc_ret
{
  unsigned int x;
  unsigned int y;
};

/* Width-correct nondet helpers. ESBMC's symex recognises any function
 * named `nondet_*` and lowers a call to NONDET(<return-type>), so the
 * signature alone determines the resulting bitvector width.
 *
 * Why these matter: `(uint256_t)nondet_uint()` zero-extends a 32-bit
 * nondet to 256 bits, silently constraining the value to [0, 2^32).
 * That broke any --bound test involving real ETH magnitudes (1 ether
 * = 10^18 > 2^32) — paths needing balances ≥ 1 ether became
 * unsatisfiable and assertions held vacuously. Use these helpers
 * everywhere a true unconstrained nondet of the matching width is
 * needed (msg_value / balances / block.* / address fields). */
uint256_t nondet_uint256();
address_t nondet_address_t();

/* Bounded nondet initial contract balance in [0, 2^128). See
 * solidity_misc.c for the rationale (avoids the near-2^256 overflow corner
 * that spuriously drops reentrant `.call{value:}` callbacks). */
uint256_t _ESBMC_nondet_init_balance();

/* T1.1 Stage S2: hash-fold of (address, index) into a 64-bit dyn-array
 * slot key for per-instance element addressing.  See solidity_array.c. */
uint64_t _ESBMC_dynarr_idx(address_t addr, uint256_t idx);

#endif /* SOLIDITY_TYPES_H */
