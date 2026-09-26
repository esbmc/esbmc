/* Fixed-point shifts with constant and runtime amounts, pinned by native
 * execution. */
#include <assert.h>

int main(void)
{
  short _Fract sh = 0.25hr;
  assert(sh << 1 == 0.5hr);
  assert(sh >> 1 == 0.125hr);

  int n = 2;
  assert((sh << n) == (sh << 2));
  assert((sh >> n) == 0.0625hr);
  return 0;
}
