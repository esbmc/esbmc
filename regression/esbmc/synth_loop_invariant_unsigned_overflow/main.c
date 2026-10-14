/* The same loop synth_loop_invariant_overflow_ok admits under --overflow-check,
 * run under --unsigned-overflow-check instead. That flag makes goto_check
 * instrument unsigned arithmetic too, so the synthesised `s0 + (i - i0) * e`
 * draws unsigned-overflow claims on operations the user never wrote -- there is
 * no integer type left to emit the closed form at. Must decline, and the user's
 * own verdict must be unaffected. */
#include <assert.h>
unsigned int nondet_uint();
int main(void)
{
  unsigned int n = nondet_uint();
  __ESBMC_assume(n <= 8);
  unsigned int i = 0, s = 0;
  while (i < n)
  {
    s = s + 2;
    i = i + 1;
  }
  assert(s == 2 * n);
  return 0;
}
