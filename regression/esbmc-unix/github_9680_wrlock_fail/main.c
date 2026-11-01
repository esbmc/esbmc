/* #9680: the other thread must be able to run between wrlock and x = 1. */
#include <assert.h>
#include <pthread.h>
pthread_rwlock_t l;
int x;
void *t(void *a)
{
  if (pthread_rwlock_tryrdlock(&l) != 0 && x == 0)
    assert(0);
  return 0;
}
int main(void)
{
  pthread_rwlock_init(&l, 0);
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_rwlock_wrlock(&l);
  x = 1;
  pthread_rwlock_unlock(&l);
  pthread_join(th, 0);
  return 0;
}
