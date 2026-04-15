/* Atomic load (CK_AtomicToNonAtomic): one thread reads flag into a local.
 * Another thread writes flag.  Neither access should be a data race.
 * Exercises the RHS-only-atomic path. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_uint flag = 0;

void *reader(void *arg)
{
  unsigned int v = flag; /* CK_AtomicToNonAtomic */
  (void)v;
  return NULL;
}

void *writer(void *arg)
{
  flag = 1; /* CK_NonAtomicToAtomic */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, reader, NULL);
  pthread_create(&t2, NULL, writer, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  return 0;
}
