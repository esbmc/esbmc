#include <stdio.h>
#include <string.h>

char global_arr[10];

char nondet_char();

void nondet_init_arr(char *arr)
{
  unsigned int i = 0;
  for(; i < 9; i++)
    arr[i] = nondet_char();
}

void global_fun()
{
  nondet_init_arr(global_arr);
  unsigned int len = strlen(global_arr);
  __ESBMC_assert(len == 0, "Error: array must be empty");
}

int main()
{
  global_fun();
  return 0;
}
