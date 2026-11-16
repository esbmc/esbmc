// The loop moves s by k, an integer it assigns before every read. The step
// must still havoc a's object and keep s inside it, or the writes through s
// reach nothing and the step proves a[0] == prev.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  unsigned char *a = malloc(4);
  unsigned char *s = a;
  unsigned i = 0;
  for (;;)
  {
    unsigned char prev = a[0];
    *s = *s + 1;
    size_t k = 0;
    s = s + k;
    if (++i > 3)
      __VERIFIER_assert(a[0] == prev);
  }
}
