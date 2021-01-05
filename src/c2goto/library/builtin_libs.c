/*******************************************************************\

Module: Builtin Models

Author: Rafael Menezes

Description: Models for custom functions that are easier to implement
 and test here

\*******************************************************************/

#include "../headers/buitin_libs.hs"

extern void __ESBMC_atomic_begin();
extern void __ESBMC_atomic_end();

int __ESBMC_sync_fetch_and_add(int *ptr, int value)
{
  __ESBMC_atomic_begin();
  int initial = *ptr;
  *ptr += value;
  __ESBMC_atomic_end();
  return initial;
}