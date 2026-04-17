/* _Atomic pre-increment (++counter) is a sequentially-consistent RMW:
 * two threads each doing ++counter must always produce counter == 2. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int counter = 0;

void *increment(void *arg)
{
  ++counter; /* atomic RMW pre-increment */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, increment, NULL);
  pthread_create(&t2, NULL, increment, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  assert(counter == 2); /* must hold: RMW prevents lost update */
  return 0;
}
