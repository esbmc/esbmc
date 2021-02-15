#include <string.h>
#include <assert.h>

char global_arr[10];

void nondet_arr(char *arr)
{
  unsigned int i = 0;
  for(; i < 10; i++)
    arr[i] = 'A';
  arr[9] = '\0';
}

void global_fun()
{
  nondet_arr(global_arr);
  int len = strlen(global_arr);
  assert(len == 0);
}

int main()
{
  global_fun();
  return 0;
}
