/* Solidity miscellaneous: min/max, reentrancy check, state initialization */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "solidity_types.h"

unsigned int nondet_uint();

extern uint256_t msg_data;
extern address_t msg_sender;
extern uint32_t msg_sig;
extern uint256_t msg_value;
extern address_t _ESBMC_enclosing_contract_address;
extern void *_ESBMC_enclosing_contract_this;
extern uint256_t tx_gasprice;
extern address_t tx_origin;
extern uint256_t block_basefee;
extern uint256_t block_blobbasefee;
extern uint256_t block_chainid;
extern address_t block_coinbase;
extern uint256_t block_difficulty;
extern uint256_t block_gaslimit;
extern uint256_t block_number;
extern uint256_t block_prevrandao;
extern uint256_t block_timestamp;
extern unsigned int _gaslimit;
extern unsigned int sol_max_cnt;
extern unsigned int sol_eoa_max_cnt;
extern unsigned int esbmc_array_count;

/* Bounded nondet initial contract balance: a value in [0, 2^128).
 * A full uint256 nondet would let the solver pick a value near 2^256 so
 * that a later deposit (`$balance += value`) wraps around to a tiny value,
 * spuriously failing the `.call{value:}` funding check and silently
 * dropping reentrant callbacks (a completeness gap that hides reentrancy
 * bugs in differential harnesses). Real ETH balances are far below 2^128,
 * so this removes only physically-impossible states while preserving every
 * balance-positive path (e.g. `require(address(this).balance >= V)`). */
uint256_t _ESBMC_nondet_init_balance(void)
{
__ESBMC_HIDE:;
  uint256_t b = nondet_uint256();
  __ESBMC_assume(b < ((uint256_t)1 << 128));
  return b;
}

uint256_t _max(unsigned int bitwidth, bool is_signed)
{
__ESBMC_HIDE:;
  __ESBMC_assume(bitwidth > 0 && bitwidth <= 256);
  if (is_signed)
  {
    return ((uint256_t)1 << (bitwidth - 1)) - (uint256_t)1;
  }
  else
  {
    if (bitwidth == 256)
    {
      return (uint256_t)-1;
    }
    return ((uint256_t)1 << bitwidth) - (uint256_t)1;
  }
}

int256_t _min(unsigned int bitwidth, bool is_signed)
{
__ESBMC_HIDE:;
  if (is_signed)
  {
    __ESBMC_assume(bitwidth > 1 && bitwidth <= 256);
    /* -2^(N-1), without negating 2^(N-1), which overflows for N = 256 */
    int256_t half = (int256_t)1 << (bitwidth - 2);
    return -half - half;
  }
  else
  {
    return (int256_t)0; // Min of unsigned is always 0
  }
}

/* type(C).creationCode / runtimeCode — return a value that's
 * deterministic per source contract (the frontend assigns a unique
 * uint32 id per contract+property pair via interface_id_table) and
 * pairwise-distinct across distinct contracts. `~id` is a bijection
 * on uint32, sufficient for real-EVM stability semantics: two reads
 * of `type(C).creationCode` agree, and `type(C).creationCode !=
 * type(D).creationCode`. Closes ledger #15 (the over-approx-nondet
 * model is replaced by an identity-bijective model, same family as
 * the crypto-hash abstraction).
 */
unsigned int _creationCode(uint32_t id)
{
__ESBMC_HIDE:;
  return (unsigned int)~id;
}

unsigned int _runtimeCode(uint32_t id)
{
__ESBMC_HIDE:;
  return (unsigned int)~id;
}

/* type(I).interfaceId — bytes4 deterministic per interface. Same
 * stability + distinctness contract as creationCode/runtimeCode. */
uint32_t _interfaceId(uint32_t id)
{
__ESBMC_HIDE:;
  return (uint32_t)~id;
}

void _ESBMC_check_reentrancy(const bool _ESBMC_mutex)
{
__ESBMC_HIDE:;
  if (_ESBMC_mutex)
    assert(!"Reentrancy behavior detected");
}

void initialize()
{
__ESBMC_HIDE:;
  // we assume it starts from an EOA
  msg_data = nondet_uint256();
  msg_sender = (address_t)nondet_uint();
  msg_sig = nondet_uint();
  msg_value = nondet_uint256();

  // Enclosing-contract ambient starts empty (address 0 / NULL).
  // Every contract method entry will overwrite it via the per-method
  // wrapper emitted by get_function_definition; library bodies read
  // it after the wrapper has fired in the calling contract method.
  _ESBMC_enclosing_contract_address = (address_t)0;
  _ESBMC_enclosing_contract_this = (void *)0;

  tx_gasprice = nondet_uint256();
  // this can only be an EOA's address
  tx_origin = (address_t)nondet_uint();

  block_basefee = nondet_uint256();
  block_blobbasefee = nondet_uint256();
  block_chainid = nondet_uint256();
  block_coinbase = (address_t)nondet_uint();
  block_difficulty = nondet_uint256();
  block_gaslimit = nondet_uint256();
  block_number = nondet_uint256();
  block_prevrandao = nondet_uint256();
  block_timestamp = nondet_uint256();

  _gaslimit = nondet_uint();

  sol_max_cnt = 0;
  sol_eoa_max_cnt = 0;
  esbmc_array_count = 0;
}

/* Per-tx ambient reseed. Called from the per-contract dispatcher
 * while-loop prologue (_ESBMC_Main_<C>) so each iteration models a
 * distinct transaction with its own sender / value / block context.
 * block.number and block.timestamp are constrained non-decreasing
 * (real EVM is monotone). msg_sig, block_chainid, _gaslimit are NOT
 * touched: msg_sig is set by per-method dispatch, chainid is
 * chain-constant, _gaslimit is a per-call intrinsic via gasleft().
 * The constructor's binding `owner = msg.sender` runs BEFORE the
 * first call to this helper, so the deployer identity stays stored
 * and per-iter senders are properly distinct. */
void _sol_per_tx_reseed()
{
__ESBMC_HIDE:;
  /* tx-envelope state */
  msg_data = nondet_uint256();
  msg_sender = (address_t)nondet_uint();
  msg_value = nondet_uint256();
  tx_origin = (address_t)nondet_uint();
  tx_gasprice = nondet_uint256();

  /* tx.origin vs msg.sender:
   * Real EVM:
   *   - Direct EOA → contract call: tx.origin == msg.sender.
   *   - Contract → contract relay  : tx.origin == original-EOA,
   *                                  msg.sender == calling contract,
   *                                  so tx.origin != msg.sender.
   * The bound-mode dispatcher models an arbitrary sequence of
   * top-level external calls.  Any of those calls may originate
   * from another contract that wraps the call on behalf of the EOA
   * (the SWC-115 phishing pattern: an attacker contract is invoked
   * by `owner`, and re-enters this contract with tx.origin == owner
   * but msg.sender == attacker).  Both regimes must be reachable —
   * leaving msg_sender and tx_origin independent lets the solver
   * pick either.  A user wanting to restrict to the direct-EOA case
   * can add `require(tx.origin == msg.sender);` in their harness. */

  /* block state — monotonic on number / timestamp */
  uint256_t _new_bn = nondet_uint256();
  __ESBMC_assume(_new_bn >= block_number);
  block_number = _new_bn;

  uint256_t _new_ts = nondet_uint256();
  __ESBMC_assume(_new_ts >= block_timestamp);
  block_timestamp = _new_ts;

  block_basefee = nondet_uint256();
  block_blobbasefee = nondet_uint256();
  block_coinbase = (address_t)nondet_uint();
  block_difficulty = nondet_uint256();
  block_gaslimit = nondet_uint256();
  block_prevrandao = nondet_uint256();
}

/* Revert observation: a
 * verification-only readable flag recording whether the most recent external
 * (public/external) call reverted.
 *
 * - _ESBMC_sol_reverted_flag : global flag (NOT contract state), so a revert's
 *     `*this` rollback does not reset it.  Default zero => "no revert" before
 *     any call.
 * - _ESBMC_sol_mark_revert() : set by the frontend at every captured revert
 *     site (revert / require-false / custom error) — see
 *     build_revert_rollback_block.
 * - _ESBMC_sol_clear_revert(): cleared by the frontend at every public/external
 *     function entry (the EVM call boundary).
 * - __ESBMC_reverted()        : user-facing read, hijacked from a user stub at
 *     analysis time (solidity_convert_ref.cpp is_intrinsic_alias).
 *
 * The mark/clear calls are tagged `skipped` by the frontend and their bodies
 * live in this library file, so they never contribute to condition/branch
 * coverage. */
bool _ESBMC_sol_reverted_flag = false;

void _ESBMC_sol_mark_revert(void)
{
__ESBMC_HIDE:;
  _ESBMC_sol_reverted_flag = true;
}

void _ESBMC_sol_clear_revert(void)
{
__ESBMC_HIDE:;
  _ESBMC_sol_reverted_flag = false;
}

bool __ESBMC_reverted(void)
{
__ESBMC_HIDE:;
  return _ESBMC_sol_reverted_flag;
}
