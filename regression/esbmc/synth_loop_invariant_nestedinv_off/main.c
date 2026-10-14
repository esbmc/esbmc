/* The mutation partner of synth_loop_invariant_nestedinv: the same program with
 * the outer marker removed, so nothing protects the inner loop and it is
 * synthesised. Without this the sibling's silence could come from the
 * recogniser declining for any other reason. */
#include <assert.h>

int main(void)
{
  unsigned int o = 0;
  unsigned int total = 0;

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
