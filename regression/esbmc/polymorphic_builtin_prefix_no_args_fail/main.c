#include <assert.h>

/* Counterpart of polymorphic_builtin_prefix_no_args: the call is not a
 * polymorphic builtin, so it keeps its body-less declaration and returns
 * nondet -- which the assertion must be free to violate. */
int __sync_fetch_and_add_mine(void);

int main(void)
{
  int r = __sync_fetch_and_add_mine();
  assert(r == 42);
  return 0;
}
