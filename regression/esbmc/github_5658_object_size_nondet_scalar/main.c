#include <assert.h>
#include <stddef.h>

// Companion to github_5658_object_size_nondet: that test exercises the
// internal_deref_items.empty() half of the guard in
// intrinsic_get_object_size (an unconstrained pointer parameter resolves to
// nothing). This test exercises the other half -- !is_array_type(...) --
// where the pointer DOES resolve to a concrete object, but that object is a
// scalar, not an array. Both halves share the same nondet-size fallback, so
// this must also reach a verdict instead of aborting.
int main()
{
  int x = 0;
  size_t n = __ESBMC_get_object_size((char *)&x);
  assert(n >= 0);
  return 0;
}
