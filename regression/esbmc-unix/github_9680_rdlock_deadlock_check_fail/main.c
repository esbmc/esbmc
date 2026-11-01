/* #9680: under --deadlock-check the rdlock model must also let the other
 * thread run between the rdlock and x = 1. */
#include <assert.h>
#include <pthread.h>
pthread_rwlock_t l;
int x;
void *t(void *a)
{
  if (pthread_rwlock_trywrlock(&l) == 0)
    pthread_rwlock_unlock(&l);
  else if (x == 0)
    assert(0);
  return 0;
}
int main(void)
{
  pthread_rwlock_init(&l, 0);
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_rwlock_rdlock(&l);
  x = 1;
  pthread_rwlock_unlock(&l);
  pthread_join(th, 0);
  return 0;
}
