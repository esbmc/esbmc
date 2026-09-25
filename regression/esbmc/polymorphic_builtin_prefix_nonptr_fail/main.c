#include <assert.h>

/* Counterpart of polymorphic_builtin_prefix_nonptr: declining the call leaves
 * the body-less declaration, so the result is nondet and the assertion must be
 * free to fail rather than being folded away. */
int __atomic_load_n_bogus(int);

int main(void)
{
  int r = __atomic_load_n_bogus(3);
  assert(r == 42);
  return 0;
}
