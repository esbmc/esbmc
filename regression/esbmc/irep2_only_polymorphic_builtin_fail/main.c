#include <stdatomic.h>

/* The instantiated body carries the memory-safety obligations of the access
 * the builtin performs. Body-less, the store is a no-op and the null
 * dereference is missed entirely. */
int main(void)
{
  atomic_int *p = 0;
  atomic_store(p, 1);
  return 0;
}
