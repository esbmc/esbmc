/*******************************************************************\

Module: Builtin Models

Author: Rafael Menezes

Description: Models for custom functions that are easier to implement
 and test here

\*******************************************************************/

int __ESBMC_sync_fetch_and_add(int *ptr, int value)
{
  __ESBMC_atomic_begin();
  int initial = *ptr;
  *ptr += value;
  __ESBMC_atomic_end();
  return initial;
}

int __ESBMC_parity(unsigned int value)
{ 
    unsigned counter = 0;

    while (counter > 0)
    {
      counter = value & 1 == 1 ? counter + 1 : counter;
      value = value >> 1;
    }
    return counter % 2;
} 

int __ESBMC_clz(unsigned int value)
{
  int counter, msb, i, int_size;

  int_size = sizeof(int) * 8;
  msb = 1 << (int_size - 1);
  counter = 0;

  for (i = 0; i < int_size; i++)
  {
    if ((value << i) & msb)
    {
      break;
    }
    counter ++;
  }
  return counter;
}

