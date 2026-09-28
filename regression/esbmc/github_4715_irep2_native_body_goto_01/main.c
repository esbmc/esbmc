// W1-loc spike Phase C (esbmc/esbmc#4715): code_goto2t/code_label2t on the
// native path. Covers the three jump shapes that exercise different parts of
// finish_gotos: a forward jump within one scope, a jump *out of* a nested block
// (whose label sits at a shallower destructor-stack depth, so finish_gotos must
// emit the scope-exit DEADs), and a backward jump forming a loop.
//
// Native conversion is byte-identical to the fallback by construction, so this
// test cannot detect the handlers silently ceasing to fire; it pins that the
// native path computes the right answer. Byte-identity itself is discharged by
// the flag-on/flag-off --goto-functions-only sweep described in the PR.
#include <assert.h>

int forward(int a)
{
  int x = a;
  if (x < 0)
    goto neg;
  x += 10;
  goto done;
neg:
  x = -x;
done:
  return x;
}

int scope_jump(int a)
{
  int x = a;
  {
    int y = x + 1;
    if (y > 3)
      goto out;
    x = y;
  }
out:
  return x;
}

int backward(int a)
{
  int i = 0;
top:
  i += 1;
  if (i < a)
    goto top;
  return i;
}

int main(void)
{
  assert(forward(1) == 11);
  assert(forward(-2) == 2);
  assert(scope_jump(5) == 5);
  assert(scope_jump(1) == 2);
  assert(backward(3) == 3);
  return 0;
}
