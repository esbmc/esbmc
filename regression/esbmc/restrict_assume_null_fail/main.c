#include <assert.h>

/* --restrict-assume must NOT prove this. A null pointer designates no object
 * and is never accessed here, so f(NULL, NULL) is a conforming call
 * (C11 6.7.3.1p4) and a != b does not follow from restrict. */
void f(void *restrict a, void *restrict b)
{
  assert(a != b);
}
