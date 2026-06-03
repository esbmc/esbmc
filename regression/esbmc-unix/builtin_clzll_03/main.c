#include <stdint.h>
#include <assert.h>

int main()
{
  uint64_t x;
  int clz;

  x = 0;
  __builtin_clzll(x);

  return 0;
}
