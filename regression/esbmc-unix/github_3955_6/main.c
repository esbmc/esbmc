/* _Atomic compound assignment (+=) is a sequentially-consistent RMW:
 * the load and store are indivisible, so two threads each doing
 * counter += 1 must always produce counter == 2 (no lost update).
 * Contrasts with github_3955_3 where counter = counter + 1 can lose an
 * update because load and store are two separate atomic operations. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int counter = 0;

void *increment(void *arg)
{
  counter += 1; /* atomic RMW — single indivisible operation */
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
