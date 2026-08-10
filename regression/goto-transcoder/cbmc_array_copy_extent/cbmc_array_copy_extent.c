#include <assert.h>

void __CPROVER_array_copy(void *, const void *);

int main(void)
{
  int s[5] = {7, 7, 7, 7, 7};
  int d[3] = {0, 0, 0};
  // Extents differ, so this is not a whole-array assignment: CBMC leaves a
  // longer destination unconstrained past the source extent, and the
  // array_replace counterpart preserves the tail instead. Declined rather
  // than approximated in either direction.
  __CPROVER_array_copy(d, s);
  assert(d[2] == 7);
  return 0;
}
