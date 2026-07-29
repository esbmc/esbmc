#include <assert.h>
#include <stdatomic.h>

/* Every C11 <stdatomic.h> generic function lowers to a __c11_atomic_*
 * builtin. Before the fix these were body-less: stores were dropped and
 * loads returned a fresh nondeterministic value. See issue #2174. */
int main(void)
{
  atomic_int a;

  atomic_init(&a, 10);
  assert(atomic_load(&a) == 10);

  assert(atomic_exchange(&a, 20) == 10);
  assert(atomic_load(&a) == 20);

  assert(atomic_fetch_add(&a, 5) == 20);
  assert(atomic_load(&a) == 25);

  assert(atomic_fetch_sub(&a, 3) == 25);
  assert(atomic_load(&a) == 22);

  atomic_store(&a, 0xF0);
  assert(atomic_fetch_or(&a, 0x0F) == 0xF0);
  assert(atomic_load(&a) == 0xFF);

  assert(atomic_fetch_and(&a, 0x3C) == 0xFF);
  assert(atomic_load(&a) == 0x3C);

  assert(atomic_fetch_xor(&a, 0xFF) == 0x3C);
  assert(atomic_load(&a) == 0xC3);

  int expected = 0xC3;
  assert(atomic_compare_exchange_strong(&a, &expected, 99));
  assert(atomic_load(&a) == 99);

  /* A failing CAS writes the current value back into `expected`. */
  int stale = 5;
  assert(!atomic_compare_exchange_strong(&a, &stale, 7));
  assert(stale == 99);
  assert(atomic_load(&a) == 99);

  return 0;
}
