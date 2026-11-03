#include <pthread.h>

/* #8189: main releases the lock through a pointer, so t1 can take it while t2
 * writes. */
int global;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_t id1, id2;

void *t1(void *arg)
{
  pthread_mutex_lock(&m);
  global++;
  pthread_mutex_unlock(&m);
  return 0;
}

int (*unlock_fn)(pthread_mutex_t *) = pthread_mutex_unlock;

void *t2(void *arg)
{
  global++;
  return 0;
}

int main()
{
  pthread_create(&id1, 0, t1, 0);
  pthread_mutex_lock(&m);
  pthread_create(&id2, 0, t2, 0);
  unlock_fn(&m);
  pthread_join(id2, 0);
  return 0;
}
