// W1-loc spike Phase C (esbmc/esbmc#4715): the targets rollback is needed even
// with no goto/label statement in sight. A GCC statement expression is lowered
// by remove_sideeffects re-entering the *legacy* convert(), which registers its
// labels and gotos in the shared targets struct. The native dispatcher never
// sees them, so it cannot clear them field by field -- convert_function must
// restore targets wholesale when the attempt is abandoned.
//
// The trailing array declaration is what abandons it: arrays are excluded from
// the native decl handler (possible VLA), so f() falls back *after* the
// statement expression already registered a label pointing into the discarded
// native program. Without the rollback finish_gotos aborts on the dangling
// entry ("finish_gotos: unexpected goto").
#include <assert.h>

int g;

void f(void)
{
  ({
  lbl:
    g++;
    if (g < 3)
      goto lbl;
    g;
  });
  int arr[3];
  arr[0] = 1;
}

int main(void)
{
  f();
  assert(g == 3);
  return 0;
}
