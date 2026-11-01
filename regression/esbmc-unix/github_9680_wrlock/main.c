/* #9680: x is 0 whenever the lock is free, so a successful tryrdlock in the
 * other thread, which may now run right after the wrlock, always reads 0. */
#include <assert.h>
#include <pthread.h>
pthread_rwlock_t l;
int x;
void *t(void *a)
{
  if (pthread_rwlock_tryrdlock(&l) == 0)
  {
    assert(x == 0);
    pthread_rwlock_unlock(&l);
  }
  return 0;
}
int main(void)
{
  pthread_rwlock_init(&l, 0);
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_rwlock_wrlock(&l);
  x = 1;
  x = 0;
  pthread_rwlock_unlock(&l);
  pthread_join(th, 0);
  return 0;
}
