/* Two writes to the same accumulator in one iteration. The per-iteration
 * addend is then not the single value the closed form assumes, so the
 * recogniser declines rather than summarise only the first write. */
#include <assert.h>

int main(void)
{
  unsigned int i = 0;
  unsigned int s = 0;

  while (i < 4)
  {
    s = s + 1;
    s = s + 2;
    i = i + 1;
  }

  assert(s == 12);
  return 0;
}
