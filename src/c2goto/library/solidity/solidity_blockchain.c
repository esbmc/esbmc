/*
 * Solidity block, transaction, and message context variables & functions.
 *
 * All properties are modelled as unconstrained nondeterministic values,
 * which is sound (over-approximate) for safety verification.
 */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "solidity_types.h"

/* ── msg variables ─────────────────────────────────────────────── */
uint256_t msg_data;
address_t msg_sender;
__uint32_t msg_sig;
uint256_t msg_value;

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
