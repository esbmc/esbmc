/* Atomic store (CK_NonAtomicToAtomic): one thread writes flag = 1.
 * The write must not be flagged as a data race, and after join the
 * value must be correct.  Exercises the LHS-only-atomic path. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_uint flag = 0;

void *writer(void *arg)
{
  flag = 1; /* CK_NonAtomicToAtomic */
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, writer, NULL);
  pthread_join(t, NULL);
  unsigned int v = flag; /* CK_AtomicToNonAtomic */
  assert(v == 1);
  return 0;
}
