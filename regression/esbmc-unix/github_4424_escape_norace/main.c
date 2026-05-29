// Companion to github_4424_escape_race: the address of the stack local still
// escapes to the worker thread, but now both accesses are guarded by the SAME
// mutex, so they are mutually exclusive and there is no data race.  This
// confirms that treating an address-taken local as race-eligible does not
// raise a spurious race when the program is correctly synchronised.
//
// Expected: VERIFICATION SUCCESSFUL.
#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  int *p = (int *)arg;
  pthread_mutex_lock(&mutex);
  (*p)++;
  pthread_mutex_unlock(&mutex);
  return NULL;
}

int main(void)
{
  pthread_t id;
  int i = 0;
  pthread_create(&id, NULL, t_fun, (void *)&i);
  pthread_mutex_lock(&mutex);
  i++;
  pthread_mutex_unlock(&mutex);
  pthread_join(id, NULL);
  return 0;
}
