/* #8653: with --no-pointer-check, memcpy must not claim a code-object
   dereference failure when the pointer may target a function. */
#include <assert.h>
#include <string.h>

void f(void)
{
}

int nondet_int(void);

int main(void)
{
  char buf[4] = {1, 2, 3, 4};
  char dst[4] = {0};
  const void *p = nondet_int() ? (const void *)f : (const void *)buf;
  memcpy(dst, p, sizeof dst);
  if (p == (const void *)buf)
    assert(dst[0] == 0);
  return 0;
}
