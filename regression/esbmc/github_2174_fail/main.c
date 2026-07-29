#include <assert.h>
#include <stdatomic.h>

/* Negative counterpart of github_2174: the value stored must be the value
 * loaded, so this assertion has to be reported as violated. A body-less
 * __c11_atomic_load would return nondet and could satisfy it. */
int main(void)
{
  atomic_int a;

  atomic_init(&a, 0);
  atomic_store(&a, 42);
  assert(atomic_load(&a) == 7);

  return 0;
}
