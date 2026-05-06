#include <assert.h>

/* Callee declares two parameters but the call site (via an unprototyped
 * function pointer) only supplies one.  Per C17 6.5.2.2/9 this is undefined
 * behaviour; ESBMC models the missing argument as nondet, so the assertion
 * on the supplied argument still holds. */
void g(int x, int y)
{
  assert(x == 5);
}

void trampo(void (*fp)())
{
  fp(5);
}

int main()
{
  trampo(g);
}
