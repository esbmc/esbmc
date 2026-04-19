#include <string.h>

int main()
{
  const char *str = "Hello";
  const size_t len = strlen(str);  // len = 5
  int *ptr = (int*)len;  // Use const length as address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
