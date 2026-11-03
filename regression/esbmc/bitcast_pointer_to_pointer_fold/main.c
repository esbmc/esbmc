/* A pointer read through a pointer to another pointer type is a bitcast
 * between two pointer types. It still names the object it pointed at, so
 * both claims fold without the solver. */
#include <stddef.h>

int x;

int main()
{
  int *p = &x;
  void *q = *(void **)&p;
  __ESBMC_assert(q != NULL, "q is not NULL");
  __ESBMC_assert(__ESBMC_same_object(q, &x), "q points at x");
  return 0;
}
