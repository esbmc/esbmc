// Pins inner-loop summary support inside Shape B IF/else branches.
//
// The outer loop's body is `if (cond) { inner1 } else { inner2 }` —
// each branch contains an entire inner loop. Phase 2b's outer-body
// inner-loop summary block handled the case where the inner sits at
// the top level of the outer body, but the Shape B IF/else arms used
// `collect_straight_line` which couldn't process inner-loop heads.
//
// The fix threads `loop_skip` and `deref_map` into
// `collect_straight_line` and recognises the conservative monotone-
// counter shape inline inside arms, emitting the same refined
// `pre := v; v := ite(...)` assigns. The arm-end scan also learns to
// skip past inner-loop back-edges so it doesn't truncate the arm at
// the inner's head IF.
//
// Pure-havoc fallback is intentionally NOT supported in arms: it
// would emit a `summary_cond` that a flat arm-assigns vector can't
// carry. If the inner isn't refined-eligible, the arm collection
// fails and we fall back to today's UNKNOWN.
//
// Lifted from termination-memory-alloca/b.18-alloca.
#include <stdlib.h>
#include <alloca.h>
extern int __VERIFIER_nondet_int(void);

int test_fun(int x, int y)
{
  int *x_ref = (int *)alloca(sizeof(int));
  int *y_ref = (int *)alloca(sizeof(int));
  *x_ref = x;
  *y_ref = y;
  while ((*x_ref > 0) && (*y_ref > 0))
  {
    if (*x_ref > *y_ref)
    {
      while (*x_ref > 0)
      {
        *x_ref = *x_ref - 1;
      }
    }
    else
    {
      while (*y_ref > 0)
      {
        *y_ref = *y_ref - 1;
      }
    }
  }
  if (*x_ref < 0 && *y_ref < 0)
    return 0;
  return *x_ref + *y_ref;
}

int main(void)
{
  return test_fun(__VERIFIER_nondet_int(), __VERIFIER_nondet_int());
}
