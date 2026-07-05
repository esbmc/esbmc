#include <assert.h>
#include <stddef.h>

// Regression for #5658: __ESBMC_get_object_size used to abort() with
// "cannot determine the size of a non-array object" whenever the pointer
// could not be resolved to a concrete array (e.g. an unconstrained pointer
// parameter, as produced by the Python frontend for a symbolic `bytes`
// argument). It must now model the size as an unconstrained nondet value
// instead of aborting, so callers still reach a verdict.
size_t f(char *p)
{
  size_t n = __ESBMC_get_object_size(p);
  assert(n >= 0);
  return n;
}

int main()
{
  return 0;
}
