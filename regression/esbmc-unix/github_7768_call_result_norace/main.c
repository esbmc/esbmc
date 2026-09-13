#include <pthread.h>

/* No-race companion to github_7768_call_result_race: the increment is locked.
 * Each thread's copy of the call result is thread-local, so the two stores
 * must not be reported as racing with each other. */
int shared;
int r1, r2;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

int bump(void)
{
  pthread_mutex_lock(&m);
  shared = shared + 1;
  pthread_mutex_unlock(&m);
  return 0;
}

void *t1(void *arg)
{
  r1 = bump();
  return 0;
}

void *t2(void *arg)
{
  r2 = bump();
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  return 0;
}
