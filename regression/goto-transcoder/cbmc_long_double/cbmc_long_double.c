#include <assert.h>
int main()
{
  long double d = 1.5L;
  // Before the fix the 128-bit constant was left as raw hex and misdecoded,
  // so d read as ~0 and `d > 1.0L` was a false FAILED.
  assert(d == 1.5L);
  assert(d > 1.0L);
  assert(d * 2.0L == 3.0L);
  return 0;
}
