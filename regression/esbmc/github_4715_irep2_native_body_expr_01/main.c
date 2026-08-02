// Exercises the --irep2-native-body IREP2-native code_expression2t dispatcher
// (W1-loc spike Phase C, esbmc/esbmc#4715): the decl-free bodies of touch() and
// discard() contain side-effect-free expression statements (`x;`, `(void)a;`,
// `b;`) that goto_convert consumes natively as OTHER instructions (code2 stored
// directly, no legacy round-trip), alongside the native ASSIGN in touch();
// main() (locals + asserts) falls back to goto_convert_rec. The no-op OTHER must
// not corrupt the surrounding assignment, and verdict/GOTO must match a run
// without the flag.
#include <assert.h>

int g;

void touch(int x)
{
  x;
  g = x + 1;
}

void discard(int a, int b)
{
  (void)a;
  b;
}

int main(void)
{
  touch(5);
  assert(g == 6);

  discard(1, 2);
  return 0;
}
