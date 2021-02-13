#include <stdio.h>
#include <string.h>

char global_arr[10];


int nondet_init_arr(char *arr)
{
  unsigned int i = 0;
  for(; i < 9; i++)
    arr[i] = nondet_char();
  return 0;
}

void global_fun()
{
  int foo = nondet_init_arr(global_arr); // forcing a lhs 
  unsigned int len = strlen(global_arr);
  assert(foo == 0);
}

int main()
{
  global_fun();
  return 0;
}

