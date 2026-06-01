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
address_t _ESBMC_get_unique_address(void *obj, const char *cname)
{
__ESBMC_HIDE:;
  // __ESBMC_assume(obj != NULL);
  address_t tmp;
  do
  {
    tmp = (address_t)nondet_uint();
    if (tmp == (address_t)0)
      continue;
    if (sol_max_cnt == 0)
      break;
  } while (_ESBMC_get_addr_array_idx(tmp) == -1);

  update_addr_obj(tmp, obj, cname);
  return tmp;
}
const char *_ESBMC_get_nondet_cont_name(const char *c_array[], unsigned int len)
{
__ESBMC_HIDE:;
  unsigned int rand = nondet_uint() % len;
  return c_array[rand];
}
