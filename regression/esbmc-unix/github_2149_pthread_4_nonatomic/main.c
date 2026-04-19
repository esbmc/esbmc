/* Same pattern as github_2149_pthread_4 but with plain unsigned int instead
 * of atomic_uint.  A W/W data race MUST be detected here, confirming that
 * the #atomic suppression in rw_set only applies to _Atomic variables. */
#include <pthread.h>

unsigned int nondet_uint();

unsigned int counter = 0; /* plain, not _Atomic */

void *assign(void *arg)
{
  unsigned int v = nondet_uint();
  __ESBMC_assume(v >= 1 && v <= 100);
  counter = v;
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, assign, NULL);
  pthread_create(&t2, NULL, assign, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  return 0;
}
