#include <assert.h>

/* Counterpart of polymorphic_builtin_prefix_arity: declining the call leaves
 * the body-less declaration, so the result is nondet and the assertion must be
 * free to fail rather than being folded away. */
int __builtin_add_overflow_mine(int *p);

int main(void)
{
  int x = 0;
  int a = __builtin_add_overflow_mine(&x);
  assert(a == 42);
  return 0;
}
