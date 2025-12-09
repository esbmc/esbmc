#include <stdbool.h>

int main()
{
  const bool flag = true;  // true = 1
  int *ptr = (int*)(unsigned long)flag;  // Use const bool as address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
