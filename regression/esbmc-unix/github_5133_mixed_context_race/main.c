#include <pthread.h>

/* Soundness guard for the esbmc/esbmc#5133 fix (always-atomic classification).
 *
 * The helper `touch` reads the shared global `data`. It is reached from TWO
 * contexts: once atomically (via __VERIFIER_atomic_wrap) and once NON-atomically
 * (called directly from reader()). Because at least one call site is
 * non-atomic, `touch` must NOT be classified as always-atomic: its non-atomic
 * read still races with writer()'s concurrent non-atomic write `data = 1`.
 * The expected verdict is therefore VERIFICATION FAILED.
 *
 * This pins the soundness of compute_always_atomic_functions: suppressing the
 * race check for a function that has any non-atomic call site would hide a real
 * race -- a false negative on a racy program.
 */

int data = 0;

int touch(void)
{
  return data; /* shared read */
}

void __VERIFIER_atomic_wrap(int *out)
{
  *out = touch(); /* atomic call site */
}

void *writer(void *arg)
{
  (void)arg;
  data = 1; /* NON-atomic write -- races with the non-atomic read below */
  return 0;
}

void *reader(void *arg)
{
  (void)arg;
  int a, b;
  __VERIFIER_atomic_wrap(&a); /* atomic call site of touch */
  b = touch();                /* NON-atomic call site of touch -- racy read */
  (void)a;
  (void)b;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  reader(0);
  return 0;
}
