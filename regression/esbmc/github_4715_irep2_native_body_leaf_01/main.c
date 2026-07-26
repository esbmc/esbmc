// Exercises the --irep2-native-body IREP2-native leaf dispatcher (W1-loc spike
// Phase C, esbmc/esbmc#4715): nop()'s skip-only body and noop()'s empty body
// are consumed natively by goto_convert (code_block2t/code_skip2t read
// directly, no legacy round-trip), while main() (decl + assert) falls back to
// goto_convert_rec. Verdict and GOTO must match a run without the flag.
#include <assert.h>

void nop(void)
{
  ;
}

void noop(void)
{
}

int main(void)
{
  nop();
  noop();
  int x = 2;
  assert(x == 2);
  return 0;
}
