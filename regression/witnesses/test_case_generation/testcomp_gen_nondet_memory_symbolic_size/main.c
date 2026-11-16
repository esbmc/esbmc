#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

extern void __VERIFIER_nondet_memory(void *ptr, size_t size);
extern unsigned int __VERIFIER_nondet_uint(void);

int main()
{
  /* A buffer of nondet size has no constant byte count, so its bytes are not
   * emitted and the test case is incomplete (only n); generating it must
   * still not abort. */
  unsigned int n = __VERIFIER_nondet_uint();
  if (n == 0 || n > 5)
    return 0;
  char *b = malloc(n);
  if (!b)
    return 0;
  __VERIFIER_nondet_memory(b, n);
  if (n == 3 && b[0] == 'A')
    assert(0);
  free(b);
  return 0;
}
