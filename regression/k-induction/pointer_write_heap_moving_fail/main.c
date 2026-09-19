// The loop writes a heap array through a moving pointer, which neither a
// named object nor *p covers, so the inductive step is disabled and the base
// case finds the bug.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int *a = malloc(4 * sizeof(int));
  unsigned k = 0;
  a[0] = a[1] = a[2] = a[3] = 0;
  for (;;)
  {
    *(a + k % 4) = *(a + k % 4) + 1;
    k++;
    __VERIFIER_assert(a[0] < 3);
  }
}
