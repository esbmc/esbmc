#include <pthread.h>

/* No-race companion to github_7768_funptr_call_race: the increment is locked. */
int shared;
void (*fp)(void);
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

void bump(void)
{
  pthread_mutex_lock(&m);
  shared = shared + 1;
  pthread_mutex_unlock(&m);
}

void *t(void *arg)
{
  fp();
  return 0;
}

int main(void)
{
  fp = bump;
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  return 0;
}
