#include <assert.h>
#include <stdatomic.h>

/* --clang-c-irep2-adjust-only replaces clang_c_adjust rather than shadowing
 * it, so the C11 atomic builtins are instantiated only if the IREP2 pass
 * declares them too. Unported, clang's body-less declarations survive: stores
 * are dropped and loads return nondet. */
int main(void)
{
  atomic_int a;

  atomic_init(&a, 10);
  assert(atomic_load(&a) == 10);

  atomic_store(&a, 7);
  assert(atomic_fetch_add(&a, 5) == 7);
  assert(atomic_load(&a) == 12);

  return 0;
}
