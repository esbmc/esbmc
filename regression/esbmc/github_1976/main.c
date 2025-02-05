#include <assert.h>

//extern void* memcpy(void* dest, const void* src, __SIZE_TYPE__ n);

int main()
{
  int src = 1;
  int dest = 2;
  __builtin_memcpy(&dest, &src, sizeof(int));
  assert(dest == 1);
  int foo[44];
  __builtin_memcpy(&foo, &foo, 4);
}