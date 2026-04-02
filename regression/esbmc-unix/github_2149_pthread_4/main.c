/* Two threads write nondet values in [1, 100] to an atomic_uint.
 * The post-join assertion counter == 0 is always false (counter >= 1),
 * so this must fail on the assertion — NOT on a data race. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

unsigned int nondet_uint();

atomic_uint counter = 0;

void *assign(void *arg)
{
  unsigned int v = nondet_uint();
  __ESBMC_assume(v >= 1 && v <= 100);
  counter = v; /* CK_NonAtomicToAtomic */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, assign, NULL);
  pthread_create(&t2, NULL, assign, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  assert(counter == 0);
  return 0;
}
