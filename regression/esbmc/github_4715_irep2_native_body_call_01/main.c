// Exercises the --irep2-native-body IREP2-native code_function_call2t
// dispatcher on the C frontend (W1-loc spike Phase C, esbmc/esbmc#4715):
// bump()/reset()'s bodies are side-effect-free assignments, and each call
// from bump_twice()/run() is a bare "foo();" statement (return value
// unused, plain named callee, side-effect-free arguments), so goto_convert
// consumes each as a single FUNCTION_CALL with no legacy round-trip. The
// return-value-unused requirement means do_function_call's temp-symbol
// machinery is never entered by this kind. main() (a call whose result IS
// used, plus asserts) falls back to goto_convert_rec.
#include <assert.h>

int g;

void bump(int x)
{
  g = g + x;
}

void reset(void)
{
  g = 0;
}

void bump_twice(int x, int y)
{
  bump(x);
  bump(y);
}

int run(void)
{
  reset();
  bump_twice(2, 3);
  return g;
}

int main(void)
{
  assert(run() == 5);
  return 0;
}
