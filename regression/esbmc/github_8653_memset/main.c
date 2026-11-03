/* #8653: with --no-pointer-check, memset must not claim a code-object
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
  void *p = nondet_int() ? (void *)f : (void *)buf;
  memset(p, 0, sizeof buf);
  if (p == (void *)buf)
    assert(buf[0] == 0);
  return 0;
}
