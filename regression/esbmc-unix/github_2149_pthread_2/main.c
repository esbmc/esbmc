/* One thread only reads an atomic_uint (CK_AtomicToNonAtomic only),
 * another only writes (CK_NonAtomicToAtomic only).  Neither should
 * be flagged as a data race by --data-races-check. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_uint flag = 0;

void *reader(void *arg)
{
  unsigned v = flag; /* CK_AtomicToNonAtomic */
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
  assert(flag == 1);
  return 0;
}
