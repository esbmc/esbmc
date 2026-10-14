// Regression for issue #4424: the address of a stack local in main escapes to
// a worker thread via pthread_create.  The thread updates *p under mutex1 while
// main updates the same object by name under mutex2 — different locks, so the
// two writes race.  ESBMC used to miss this because a direct (non-dereferenced)
// access to a non-static local was dropped from the race read/write set, so
// only the thread's *p side registered on the shared address.
//
// Ground truth (SV-COMP c/goblint-regression/04-mutex_45-escape_rc): FALSE
// (the race exists), so ESBMC must report VERIFICATION FAILED.
#include <pthread.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  int *p = (int *)arg;
  pthread_mutex_lock(&mutex1);
  (*p)++; // RACE!
  pthread_mutex_unlock(&mutex1);
  return NULL;
}

int main(void)
{
  pthread_t id;
  int i = 0;
  pthread_create(&id, NULL, t_fun, (void *)&i);
  pthread_mutex_lock(&mutex2);
  i++; // RACE! different lock than the thread
  pthread_mutex_unlock(&mutex2);
  pthread_join(id, NULL);
  return 0;
}
