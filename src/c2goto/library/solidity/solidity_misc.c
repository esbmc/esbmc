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
extern unsigned int esbmc_array_count;

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
    __ESBMC_assume(bitwidth > 0 && bitwidth <= 256);
    return -((int256_t)1 << (bitwidth - 1)); // -2^(N-1)
  }
  else
  {
    return (int256_t)0; // Min of unsigned is always 0
  }
}

unsigned int _creationCode()
{
__ESBMC_HIDE:;
  return nondet_uint();
}

unsigned int _runtimeCode()
{
__ESBMC_HIDE:;
  return nondet_uint();
}

/* type(I).interfaceId — nondet over-approximation (bytes4) */
uint32_t _interfaceId()
{
__ESBMC_HIDE:;
  return (uint32_t)nondet_uint();
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
  msg_data = (uint256_t)nondet_uint();
  msg_sender = (address_t)nondet_uint();
  msg_sig = nondet_uint();
  msg_value = (uint256_t)nondet_uint();

  tx_gasprice = (uint256_t)nondet_uint();
  // this can only be an EOA's address
  tx_origin = (address_t)nondet_uint();

  block_basefee = (uint256_t)nondet_uint();
  block_blobbasefee = (uint256_t)nondet_uint();
  block_chainid = (uint256_t)nondet_uint();
  block_coinbase = (address_t)nondet_uint();
  block_difficulty = (uint256_t)nondet_uint();
  block_gaslimit = (uint256_t)nondet_uint();
  block_number = (uint256_t)nondet_uint();
  block_prevrandao = (uint256_t)nondet_uint();
  block_timestamp = (uint256_t)nondet_uint();

  _gaslimit = nondet_uint();

  sol_max_cnt = 0;
  esbmc_array_count = 0;
}
