// Exercises the --irep2-native-body IREP2-native code_ifthenelse2t dispatcher
// (W1-loc spike Phase C, esbmc/esbmc#4715): classify() has both branches
// side-effect-free, so goto_convert consumes the whole if/else natively as the
// general branch shape (v: if(!c) goto y; w: P; x: goto z; y: Q; z: ;), with no
// legacy round-trip. no_else() exercises the else-less shape (v: if(!c) goto z;
// w: P; z: ;). nested() has an if directly inside an if (no braces), so the
// outer's then-branch is itself converted natively by recursing into the
// dispatcher. main() (calls + assert) falls back to goto_convert_rec. Verdict
// and GOTO must match a run without the flag.
#include <assert.h>

int classify(int x)
{
  if (x < 0)
    return -1;
  else
    return 1;
}

int no_else(int x)
{
  int r = 0;
  if (x > 5)
    r = x - 5;
  return r;
}

int nested(int x, int y)
{
  if (x > 0)
    if (y > 0)
      return 1;
    else
      return 2;
  return 3;
}

int main(void)
{
  assert(classify(-5) == -1);
  assert(classify(5) == 1);
  assert(no_else(10) == 5);
  assert(no_else(1) == 0);
  assert(nested(1, -1) == 2);
  assert(nested(-1, -1) == 3);
  return 0;
}
