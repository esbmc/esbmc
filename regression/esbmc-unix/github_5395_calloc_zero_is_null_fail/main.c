// Regression for github #5395: calloc of a zero-sized request must track
// malloc(0) under --malloc-zero-is-null too -- i.e. return NULL.  This
// pins the option-respecting boundary: asserting the pointer is non-null
// must FAIL here (the fix routes calloc(_,0) through malloc(0), which
// returns NULL when --malloc-zero-is-null is set).
#include <stdlib.h>

void reach_error(void) {}
static void verifier_assert(int cond)
{
  if (!cond)
    reach_error();
}

int main(void)
{
  void *p = calloc(1, 0);
  verifier_assert(p != 0); // FAILS: calloc(1,0) is NULL under --malloc-zero-is-null
  return 0;
}
