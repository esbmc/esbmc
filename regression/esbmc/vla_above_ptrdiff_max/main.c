/* A VLA whose size folds to a constant above PTRDIFF_MAX cannot be bounded by
   assumption -- the assumption is identically false and would prove the whole
   program -- so the declaration is reported instead. R40, R39's principle. */
#include <assert.h>
#include <stdint.h>

int main(void)
{
  uint64_t n = 0x8000000000000000UL; /* PTRDIFF_MAX + 1 */

  char a[n];
  a[0] = 1;

  assert(a[0] == 1);
  return 0;
}
