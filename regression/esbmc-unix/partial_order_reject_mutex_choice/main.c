#include <pthread.h>
int nondet_int(void);
int x;
pthread_mutex_t m1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m2 = PTHREAD_MUTEX_INITIALIZER;
void *t(void *arg)
{
  pthread_mutex_t *l = nondet_int() ? &m1 : &m2;
  pthread_mutex_lock(l);
  x = 1;
  pthread_mutex_unlock(l);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
