#include <assert.h>

// Exercises array_convt::mk_array_symbol on the unbounded path under the array
// flattener. A nondet-size VLA gets a word_size (64-bit) index domain, so
// is_unbounded_array() is true. Computing 1UL << domain_width before the
// unbounded early-return shifted by the full 64-bit index width and was
// undefined behaviour; the element count is only needed on the bounded path.
int main()
{
  unsigned n;
  __ESBMC_assume(n > 0 && n <= 8);
  int a[n]; // nondet size -> 64-bit index domain -> unbounded array

  unsigned i;
  __ESBMC_assume(i < n);
  a[i] = 42;
  assert(a[i] == 42);
  return 0;
}
