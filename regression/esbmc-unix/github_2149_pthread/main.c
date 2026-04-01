#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_uint counter = 0;

void *increment(void *arg)
{
  counter = counter + 1; /* CK_AtomicToNonAtomic (read) + CK_NonAtomicToAtomic (write) */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, increment, NULL);
  pthread_create(&t2, NULL, increment, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  assert(counter > 0);
  return 0;
}
