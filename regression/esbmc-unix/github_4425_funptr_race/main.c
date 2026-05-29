// Regression for issue #4425: a write to a function pointer in one thread
// and an indirect call through that pointer in another thread, guarded by
// different mutexes, is a genuine data race.  ESBMC used to miss it because
// the read of the function pointer performed by the indirect call site was
// never added to the race read/write set.
//
// Ground truth (SV-COMP c/goblint-regression/04-mutex_50-funptr_rc): FALSE
// (the race exists), so ESBMC must report VERIFICATION FAILED.
#include <pthread.h>

int f1(void)
{
  return 4;
}
int f2(void)
{
  return 5;
}

int (*fp)(void) = f1;

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  pthread_mutex_lock(&mutex1);
  fp = f2; // RACE!
  pthread_mutex_unlock(&mutex1);
  return NULL;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, NULL, t_fun, NULL);
  pthread_mutex_lock(&mutex2);
  fp(); // RACE! reads fp under a different lock
  pthread_mutex_unlock(&mutex2);
  pthread_join(id, NULL);
  return 0;
}
