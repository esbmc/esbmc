// v == 0 is not inductive: v = 1 breaks it, on a path that jumps to the end
// of the loop body. The copied iteration must check it there too.
#include <assert.h>
extern int __VERIFIER_nondet_int(void);
void reach_error(void) { assert(0); }

int main(void)
{
  int v = 0;
  while (__VERIFIER_nondet_int())
  {
    if (v == 0)
      v = 1;
    else
      reach_error();
  }
  return 0;
}
