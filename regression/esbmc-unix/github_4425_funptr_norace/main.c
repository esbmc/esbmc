#include <pthread.h>

/* No-race companion to github_4425_funptr_race (esbmc/esbmc#4425).
 *
 * Identical to the racy version except that the write to `fp` and the indirect
 * call that reads `fp` are protected by the SAME mutex, so the accesses are
 * synchronized and there is no data race. The result must stay VERIFICATION
 * SUCCESSFUL: instrumenting the read of an indirect-call target must not
 * fabricate a spurious race on a correctly locked function pointer.
 */

int f1(void)
{
  return 4;
}
int f2(void)
{
  return 5;
}

int (*fp)(void) = f1;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  (void)arg;
  pthread_mutex_lock(&mutex);
  fp = f2; /* write under mutex */
  pthread_mutex_unlock(&mutex);
  return 0;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t_fun, 0);
  pthread_mutex_lock(&mutex);
  fp(); /* read of fp under the SAME mutex -> synchronized, no race */
  pthread_mutex_unlock(&mutex);
  pthread_join(id, 0);
  return 0;
}
