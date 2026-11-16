/* x = 1 precedes the acquire, so a failed try sees x == 1 (#9680). */
#include <assert.h>
#include <pthread.h>
pthread_rwlock_t l;
int x;
void *t(void *a)
{
  if (pthread_rwlock_trywrlock(&l) != 0 && x == 0)
    assert(0);
  return 0;
}
int main(void)
{
  pthread_rwlock_init(&l, 0);
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  x = 1;
  if (pthread_rwlock_tryrdlock(&l) != 0) return 0;
  pthread_rwlock_unlock(&l);
  pthread_join(th, 0);
  return 0;
}
