// Exercises the --irep2-native-body IREP2-native code_return2t dispatcher (W1-loc
// spike Phase C, esbmc/esbmc#4715): inc()/doubled() are value-returning functions
// whose bodies reduce to a side-effect-free return, so goto_convert consumes each
// code_return2t natively (a RETURN carrying the statement's own value, then an
// unconditional GOTO to the end-of-function target) with no legacy round-trip.
// doubled() additionally declares a trivial-type local before the return, so the
// block handler must reproduce convert_block's trailing-goto guard and suppress
// the local's scope-exit DEAD (dead code after the return's goto) byte-for-byte.
// main() (calls + assert) falls back to goto_convert_rec. The returned values
// must flow correctly and the verdict/GOTO must match a run without the flag.
#include <assert.h>

int inc(int p)
{
  return p + 1;
}

int doubled(int p)
{
  int t = p * 2;
  return t;
}

int main(void)
{
  int a = inc(5);
  int b = doubled(6);
  assert(a + b == 18);
  return 0;
}
