/* Same pattern as github_3955_3 but with plain int instead of _Atomic int.
 * A W/W (and R/W) data race must be detected, confirming that the #atomic
 * suppression in rw_set only applies to _Atomic variables. */
#include <assert.h>
#include <pthread.h>

int counter = 0; /* plain int, not _Atomic */

void *increment(void *arg)
{
  counter = counter + 1;
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, increment, NULL);
  pthread_create(&t2, NULL, increment, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  return 0;
}
