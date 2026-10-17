/* Solidity address management and contract object tracking */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "solidity_types.h"

unsigned int nondet_uint();

__attribute__((annotate("__ESBMC_inf_size"))) address_t sol_addr_array[1];
__attribute__((annotate("__ESBMC_inf_size"))) void *sol_obj_array[1];
__attribute__((annotate("__ESBMC_inf_size"))) const char *sol_cname_array[1];
unsigned int sol_max_cnt;

/* EOA balance map: tracks ETH balance at addresses that are NOT modelled
 * as `_ESBMC_Object_<C>` instances (i.e. EOAs and any address payable
 * value the user constructs).  Without this, `transfer/send` to an
 * unknown address only debited sender->$balance and the recipient's ETH
 * was effectively burnt — which made order-sensitive properties on
 * recipient balances vacuous (the SolidiFI TOD pattern lives here).
 *
 * Parallel arrays in the same shape as sol_addr_array so the SMT
 * encoding stays uniform; lookup is linear in sol_eoa_max_cnt and
 * therefore needs --unwind ≥ number of distinct EOAs touched. */
__attribute__((annotate("__ESBMC_inf_size"))) address_t sol_eoa_addr_array[1];
__attribute__((
  annotate("__ESBMC_inf_size"))) uint256_t sol_eoa_balance_array[1];
/* Per-address code / codehash summary arrays, sharing the EOA address
 * pool: same slot index for the same address regardless of which
 * property was read first.  First-touch initialises both fields to a
 * fresh nondet uint256; subsequent reads return that same value, so
 * `addr.codehash == addr.codehash` and `addr.code == addr.code` hold
 * within a path.  Only the fall-through (untracked-address) case
 * routes through these helpers; tracked _ESBMC_Object_<C> instances
 * keep reading their own $code / $codehash fields, which are already
 * stable per-instance from constructor time. */
__attribute__((annotate("__ESBMC_inf_size"))) uint256_t sol_eoa_code_array[1];
__attribute__((
  annotate("__ESBMC_inf_size"))) uint256_t sol_eoa_codehash_array[1];
/* Per-address `.code.length` summary. Solidity's `.code` lowers to a
 * 256-bit identity token (sol_eoa_code_array slot), but the user-
 * observable `.length` member of bytes can't reach that token, so two
 * reads of `addr.code.length` would otherwise see independent fresh
 * nondets and fail `addr.code.length == addr.code.length` invariants.
 * Parallel summary array keyed by the same EOA pool gives both reads
 * the same length per-address. */
__attribute__((
  annotate("__ESBMC_inf_size"))) uint256_t sol_eoa_code_length_array[1];
unsigned int sol_eoa_max_cnt;

int _ESBMC_get_addr_array_idx(address_t tgt)
{
__ESBMC_HIDE:;
  if (tgt == (address_t)0)
    return -1;

  for (unsigned int i = 0; i < sol_max_cnt; i++)
  {
    if ((address_t)sol_addr_array[i] == (address_t)tgt)
      return i;
  }
  return -1;
}
bool _ESBMC_cmp_cname(const char *c_1, const char *c_2)
{
__ESBMC_HIDE:;
  return c_1 == c_2;
}
void *_ESBMC_get_obj(address_t addr, const char *cname)
{
__ESBMC_HIDE:;
  int idx = _ESBMC_get_addr_array_idx(addr);
  if (idx == -1)
    // this means it's not previously stored
    return NULL;
  if (_ESBMC_cmp_cname(sol_cname_array[idx], cname))
    return sol_obj_array[idx];
  return NULL;
}
void update_addr_obj(address_t addr, void *obj, const char *cname)
{
__ESBMC_HIDE:;
  // __ESBMC_assume(obj != NULL);
  sol_addr_array[sol_max_cnt] = addr;
  sol_obj_array[sol_max_cnt] = obj;
  sol_cname_array[sol_max_cnt] = cname;
  ++sol_max_cnt;
}
/* DEFAULT (loose) variant.  Generate a nondet address and constrain
 * it to be distinct from prior allocations, via an unrolled if-chain
 * capped at 16 slots.  Loop-free, so --unwind does not truncate.
 * Trade-off: 17th allocation is unconstrained → silent under-
 * approximation beyond cap.  See README.md, section "Address
 * uniqueness modelling".  Counterpart: _ESBMC_get_unique_address_precise. */
address_t _ESBMC_get_unique_address(void *obj, const char *cname)
{
__ESBMC_HIDE:;
  address_t tmp = (address_t)nondet_uint();
  __ESBMC_assume(tmp != (address_t)0);
  if (sol_max_cnt > 0)
    __ESBMC_assume(tmp != sol_addr_array[0]);
  if (sol_max_cnt > 1)
    __ESBMC_assume(tmp != sol_addr_array[1]);
  if (sol_max_cnt > 2)
    __ESBMC_assume(tmp != sol_addr_array[2]);
  if (sol_max_cnt > 3)
    __ESBMC_assume(tmp != sol_addr_array[3]);
  if (sol_max_cnt > 4)
    __ESBMC_assume(tmp != sol_addr_array[4]);
  if (sol_max_cnt > 5)
    __ESBMC_assume(tmp != sol_addr_array[5]);
  if (sol_max_cnt > 6)
    __ESBMC_assume(tmp != sol_addr_array[6]);
  if (sol_max_cnt > 7)
    __ESBMC_assume(tmp != sol_addr_array[7]);
  if (sol_max_cnt > 8)
    __ESBMC_assume(tmp != sol_addr_array[8]);
  if (sol_max_cnt > 9)
    __ESBMC_assume(tmp != sol_addr_array[9]);
  if (sol_max_cnt > 10)
    __ESBMC_assume(tmp != sol_addr_array[10]);
  if (sol_max_cnt > 11)
    __ESBMC_assume(tmp != sol_addr_array[11]);
  if (sol_max_cnt > 12)
    __ESBMC_assume(tmp != sol_addr_array[12]);
  if (sol_max_cnt > 13)
    __ESBMC_assume(tmp != sol_addr_array[13]);
  if (sol_max_cnt > 14)
    __ESBMC_assume(tmp != sol_addr_array[14]);
  if (sol_max_cnt > 15)
    __ESBMC_assume(tmp != sol_addr_array[15]);
  update_addr_obj(tmp, obj, cname);
  return tmp;
}

/* PRECISE (sound) variant.  Same contract as the loose variant but
 * encoded as a `for`-loop linear scan over `sol_max_cnt` — no slot
 * cap.  Selected by `--solidity-precise` at the frontend; default
 * routing keeps the loose variant for regression-suite parity and
 * --unwind-coupling reasons.
 *
 * `--unwind` coupling: the loop is bounded by `sol_max_cnt`, so the
 * user must pass `--unwind N` with N >= the number of contract
 * instantiations on any path.  Without `--no-unwinding-assertions`,
 * a too-low `--unwind` produces a visible "unwinding assertion loop
 * <id>" failure, surfacing the limit explicitly.  With
 * `--no-unwinding-assertions`, the loop tail is silently truncated
 * — same blind spot as the rest of the address library
 * (_ESBMC_get_addr_array_idx is also a linear scan).
 *
 * Why not `__ESBMC_forall`?  The quantifier encoding is solver-
 * unfriendly in practice: cvc5 returns UNKNOWN on every standard
 * quantifier strategy (only `--sygus-inst` succeeds, and that
 * conflicts with cvc5 incremental mode AND is intractably slow on
 * k-induction); bitwuzla's BV-quantifier engine is also slow on
 * related patterns.  The for-loop form works uniformly across
 * bitwuzla, z3, and cvc5 — at the cost of `--unwind` coupling. */
address_t _ESBMC_get_unique_address_precise(void *obj, const char *cname)
{
__ESBMC_HIDE:;
  address_t tmp = (address_t)nondet_uint();
  __ESBMC_assume(tmp != (address_t)0);
  for (unsigned int i = 0; i < sol_max_cnt; i++)
    __ESBMC_assume(tmp != sol_addr_array[i]);
  update_addr_obj(tmp, obj, cname);
  return tmp;
}
const char *_ESBMC_get_nondet_cont_name(const char *c_array[], unsigned int len)
{
__ESBMC_HIDE:;
  unsigned int rand = nondet_uint() % len;
  return c_array[rand];
}

/* EOA balance lookup. Returns slot index or -1 if address is not yet
 * tracked. Linear scan; caller paths must --unwind at least enough to
 * cover sol_eoa_max_cnt iterations. */
int _ESBMC_eoa_get_idx(address_t addr)
{
__ESBMC_HIDE:;
  for (unsigned int i = 0; i < sol_eoa_max_cnt; i++)
  {
    if (sol_eoa_addr_array[i] == addr)
      return (int)i;
  }
  return -1;
}

/* Find or insert. On insert, the new slot's initial balance is nondet
 * (sound over-approximation: a real EOA could have any pre-existing
 * balance). User-side tests that need a deterministic starting point
 * should `require(addr.balance == 0)` (or any other constant) before
 * the first transfer.
 *
 * NB: We do NOT special-case address(0). Solidity's `transfer` reverts
 * on insufficient balance and most contracts forbid sending to 0x0 via
 * an explicit require, but the EVM itself permits it (the ETH becomes
 * unspendable). Mirroring that behaviour keeps our model conservative. */
unsigned int _ESBMC_eoa_get_or_init(address_t addr)
{
__ESBMC_HIDE:;
  int idx = _ESBMC_eoa_get_idx(addr);
  if (idx != -1)
    return (unsigned int)idx;
  unsigned int new_idx = sol_eoa_max_cnt;
  sol_eoa_addr_array[new_idx] = addr;
  sol_eoa_balance_array[new_idx] = nondet_uint256();
  sol_eoa_code_array[new_idx] = nondet_uint256();
  sol_eoa_codehash_array[new_idx] = nondet_uint256();
  sol_eoa_code_length_array[new_idx] = nondet_uint256();
  ++sol_eoa_max_cnt;
  return new_idx;
}

void _ESBMC_eoa_credit(address_t addr, uint256_t val)
{
__ESBMC_HIDE:;
  unsigned int idx = _ESBMC_eoa_get_or_init(addr);
  sol_eoa_balance_array[idx] += val;
}

uint256_t _ESBMC_eoa_balance_of(address_t addr)
{
__ESBMC_HIDE:;
  unsigned int idx = _ESBMC_eoa_get_or_init(addr);
  return sol_eoa_balance_array[idx];
}

/* Per-address `.code` summary. Returns the same 256-bit value across
 * repeated reads of the same address within a path. Only kicks in for
 * untracked addresses (tracked _ESBMC_Object_<C> instances dispatch to
 * their own $code field via get_aux_property_function). --bound only;
 * unbound mode short-circuits to fresh nondet earlier. */
uint256_t _ESBMC_code_of(address_t addr)
{
__ESBMC_HIDE:;
  unsigned int idx = _ESBMC_eoa_get_or_init(addr);
  return sol_eoa_code_array[idx];
}

/* Per-address `.codehash` summary; same shape as _ESBMC_code_of. */
uint256_t _ESBMC_codehash_of(address_t addr)
{
__ESBMC_HIDE:;
  unsigned int idx = _ESBMC_eoa_get_or_init(addr);
  return sol_eoa_codehash_array[idx];
}

/* Per-address `.code.length` summary; same shape as _ESBMC_code_of.
 * The frontend reroutes the `.length`-on-uint256-bytes fallback to
 * this helper when it can recognize the base as a `_ESBMC_code_of`
 * call — without that, two reads of `addr.code.length` produced
 * independent fresh nondets. */
uint256_t _ESBMC_code_length_of(address_t addr)
{
__ESBMC_HIDE:;
  unsigned int idx = _ESBMC_eoa_get_or_init(addr);
  return sol_eoa_code_length_array[idx];
}
