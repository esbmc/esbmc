/* [APPROX: OVER] Solidity block / transaction / message context.
 *
 * All msg.*, tx.*, block.* variables and blockhash/blobhash are
 * unconstrained nondet uint256_t / address_t. This is sound (over-
 * approximate) for safety verification: every concrete miner/attacker
 * choice is a possible realisation.
 *
 * Consequences:
 *  - No relationship between successive block.number reads (they are NOT
 *    monotonic in the model). Properties of the form
 *    `assert(block.number >= block.number_prev)` cannot be verified.
 *  - No relationship between block.timestamp and block.number.
 *  - msg.sender can be any address on every call, including contracts
 *    that should not exist yet.
 *  - gasleft() decreases monotonically via `gasConsume()` within a call,
 *    but is reset to a nondet ceiling at each entry.
 *
 * False positives: invariants over sequences of block values.
 * False negatives: none expected for safety properties.
 */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "solidity_types.h"

/* ── msg variables ─────────────────────────────────────────────── */
uint256_t msg_data;
address_t msg_sender;
uint32_t msg_sig;
uint256_t msg_value;

/* Caller-instance address for reentrant msg.sender.  In a multi-type
 * (structurally-clustered) call the method body runs on the static
 * dispatch singleton _ESBMC_Object_<T>, whose $address differs from the
 * actual new C() instance's.  The dispatcher records the calling
 * instance's $address here (save/set/restore around the body) and the
 * low-level `.call`/`$call` builders use it as the reentrant msg.sender
 * so a callback sees msg.sender == address(p).  0 means "unset" — fall
 * back to this->$address (single-type, where the body already runs on
 * the instance).  Touching only this variable (not any registered
 * $address) avoids the sol_addr_array lookup blowup an $address swap
 * would cause. */
address_t _ESBMC_caller_inst_addr = 0;

/* ── enclosing contract ambient ────────────────────────────────────
 * When execution enters a contract method (either via Harness auto-
 * dispatch or a cross-contract .call()), the wrapper saves the
 * previous values and sets these to `this->$address` / `(void*)this`.
 * Library bodies, which have no real `this` of their own in the
 * Solidity semantics (internal library calls run in the caller's
 * context), read these to recover the enclosing contract's identity
 * for msg.sender swaps and balance debits on library-scope
 * transfers.  Initialised to zero/NULL; every contract method entry
 * save/sets before use.
 */
address_t _ESBMC_enclosing_contract_address;
void *_ESBMC_enclosing_contract_this;

/* ── tx variables ──────────────────────────────────────────────── */
uint256_t tx_gasprice;
address_t tx_origin;

/* ── block variables ───────────────────────────────────────────── */
uint256_t block_basefee;
uint256_t block_blobbasefee;
uint256_t block_chainid;
address_t block_coinbase;
uint256_t block_difficulty;
uint256_t block_gaslimit;
uint256_t block_number;
uint256_t block_prevrandao;
uint256_t block_timestamp;

/* ── blockhash — nondet abstraction (over-approximate) ─────────── */
uint256_t blockhash(uint256_t x)
{
__ESBMC_HIDE:;
  uint256_t result;
  return result;
}

/* ── blobhash (EIP-4844) — nondet abstraction (over-approximate) ─ */
uint256_t blobhash(uint256_t index)
{
__ESBMC_HIDE:;
  uint256_t result;
  return result;
}

/* ── gasleft ───────────────────────────────────────────────────── */
unsigned int nondet_uint();

unsigned int _gaslimit;
void gasConsume()
{
__ESBMC_HIDE:;
  unsigned int consumed = nondet_uint();
  __ESBMC_assume(consumed > 0 && consumed <= _gaslimit);
  _gaslimit -= consumed;
}
uint256_t gasleft()
{
__ESBMC_HIDE:;
  gasConsume(); // always less
  return (uint256_t)_gaslimit;
}
