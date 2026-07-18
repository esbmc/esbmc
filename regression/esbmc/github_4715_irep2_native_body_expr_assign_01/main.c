// W1-loc spike Phase C (esbmc/esbmc#4715): exercises a plain C assignment
// statement (`x = x + b;`) on the native path. The C/C++ frontends model it as
// an expression statement wrapping a sideeffect_assign2t, not the code_assign2t
// Python emits, so before this kind landed every ordinary reassignment forced
// its whole enclosing function back to goto_convert_rec (see the note in
// github_4715_irep2_native_body_while_call_01). f() -- decls, reassignments, a
// while loop, an if/else and a value return -- now converts natively end to end.
//
// Native conversion is byte-identical to the fallback by construction, so this
// test cannot detect the handler silently ceasing to fire; it pins that the
// native path computes the right answer. Byte-identity itself is discharged by
// the flag-on/flag-off --goto-functions-only sweep described in the PR.
#include <assert.h>

int g;

int f(int a, int b)
{
  int x = a;
  x = x + b;
  g = x;
  while (x < 10)
  {
    x = x + 1;
  }
  if (x > 0)
  {
    g = x;
  }
  else
  {
    g = 0;
  }
  return g;
}

int main(void)
{
  assert(f(1, 2) == 10);
  return 0;
}
