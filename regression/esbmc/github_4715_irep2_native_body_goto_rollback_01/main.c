// W1-loc spike Phase C (esbmc/esbmc#4715): pins the targets.gotos/targets.labels
// rollback in convert_function. A native attempt that consumes a code_goto2t
// pushes an entry holding an ITERATOR INTO ITS OWN goto_programt; if a later
// statement forces the whole function back to goto_convert_rec, that program is
// discarded and the iterator dangles. finish_gotos then dereferences it after
// the fallback rebuilds the body -- a segfault, not merely stale state.
//
// --error-label makes the label handler decline (convert_label turns a matching
// label into an ASSERT(false) with property metadata the native path does not
// reproduce), so f() emits its goto natively and *then* falls back at the label,
// which is exactly the ordering that leaves the dangling entry behind.
#include <assert.h>

int g;

int f(int a)
{
  if (a > 0)
    goto ERROR;
  return 0;
ERROR:
  g = 1;
  return 1;
}

int main(void)
{
  assert(f(1) == 1);
  assert(g == 1);
  return 0;
}
