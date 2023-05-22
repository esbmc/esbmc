#include <stdlib.h>
#include <assert.h>

int main()
{
  long result = strtol("2147483648", NULL, 10); // Maximum value of signed int + 1
  assert(result == 2147483647);
  
  return 0;
}
