#include <pthread.h>

/* #8189: t1 can take the lock once main releases it, while t2 writes. */
int global;
pthread_rwlock_t l = PTHREAD_RWLOCK_INITIALIZER;
pthread_t id1, id2;

void *t1(void *arg)
{
  pthread_rwlock_wrlock(&l);
  global++;
  pthread_rwlock_unlock(&l);
  return 0;
}

void *t2(void *arg)
{
  global++;
  return 0;
}

int main()
{
  pthread_create(&id1, 0, t1, 0);
  pthread_rwlock_wrlock(&l);
  pthread_create(&id2, 0, t2, 0);
  pthread_rwlock_unlock(&l);
  return 0;
}
