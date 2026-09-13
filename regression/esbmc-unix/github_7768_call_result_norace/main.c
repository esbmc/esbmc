#include <pthread.h>

/* No-race companion to github_7768_call_result_race: the increment is locked.
 * Both threads run the same call site, so the local that carries its result
 * must be private to each thread for the stores into distinct slots to stay
 * race-free. */
int shared;
int results[2];
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

int bump(void)
{
  pthread_mutex_lock(&m);
  shared = shared + 1;
  pthread_mutex_unlock(&m);
  return 0;
}

void *t(void *arg)
{
  results[(long)arg] = bump();
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, t, (void *)0);
  pthread_create(&b, 0, t, (void *)1);
  return 0;
}
