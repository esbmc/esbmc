// Pins the fix for the inner loop's recognize_loop failing when its
// prefix walk has to cross the OUTER loop's head.
//
// Before the fix, scan_prefix_defs (walking from function entry to
// the inner's head) hit the outer's head IF as an `is_target()`
// (because the outer's back-edge jumps back to it) and bailed with
// has_invalid=true. compute_safe_derefs then couldn't establish
// alloca provenance for *x_ref / *y_ref / *c, recognize_loop on the
// inner returned false, the per-loop driver returned UNKNOWN.
//
// The fix marks every loop head in the function as a justified
// forward-walk target. On the dominator path from function entry,
// the only "extra" incoming edge to a loop head is its own back-
// edge, which fires only AFTER we've entered the loop body — so the
// forward walk's encounter with the head is the dominator path
// itself, not a merge that breaks the analysis.
//
// Lifted from termination-memory-alloca/b.14-alloca.
#include <stdlib.h>
#include <alloca.h>
extern int __VERIFIER_nondet_int(void);

int test_fun(int x, int y)
{
  int *x_ref = (int *)alloca(sizeof(int));
  int *y_ref = (int *)alloca(sizeof(int));
  int *c = (int *)alloca(sizeof(int));
  *x_ref = x;
  *y_ref = y;
  *c = 0;
  while ((*x_ref == *y_ref) && (*x_ref > 0))
  {
    while (*y_ref > 0)
    {
      *x_ref = *x_ref - 1;
      *y_ref = *y_ref - 1;
      *c = *c + 1;
    }
  }
  return *c;
}

int main(void)
{
  return test_fun(__VERIFIER_nondet_int(), __VERIFIER_nondet_int());
}
