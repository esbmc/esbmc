#include <pthread.h>

/* #8189: t1 can take the lock once main releases it, while t2 writes. */
int global;
pthread_spinlock_t s;
pthread_t id1, id2;

void *t1(void *arg)
{
  pthread_spin_lock(&s);
  global++;
  pthread_spin_unlock(&s);
  return 0;
}

void *t2(void *arg)
{
  global++;
  return 0;
}

int main()
{
  pthread_spin_init(&s, 0);
  pthread_create(&id1, 0, t1, 0);
  pthread_spin_lock(&s);
  pthread_create(&id2, 0, t2, 0);
  pthread_spin_unlock(&s);
  return 0;
}
