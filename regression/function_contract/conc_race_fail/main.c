/* conc_race_fail:
 * Two threads both write to the same global without mutex protection.
 * --data-races-check detects the W/W race.
 *
 * This is the baseline: contracts alone cannot prevent races.
 * Use this to confirm data-race detection is active before adding contracts.
 *
 * Expected: VERIFICATION FAILED (data race on global counter)
 */
#include <pthread.h>

int counter = 0;

void *thread_fn(void *arg)
{
  counter++; /* unprotected write — race */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thread_fn, NULL);
  pthread_create(&t2, NULL, thread_fn, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  return 0;
}
