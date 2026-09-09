/* A user invariant on the outer loop protects the inner one too; see
 * enclosed_by_user_invariant for why. synth_loop_invariant_nestedinv_off is the
 * same program without the outer marker, where the inner loop is synthesised. */
#include <assert.h>

int main(void)
{
  unsigned int o = 0;
  unsigned int total = 0;

  __ESBMC_loop_invariant(o <= 2 && total == 4 * o);
  while (o < 2)
  {
    unsigned int i = 0;
    unsigned int s = 0;
    while (i < 4)
    {
      s = s + 1;
      i = i + 1;
    }
    total = total + s;
    o = o + 1;
  }

  assert(total == 8);
  return 0;
}
