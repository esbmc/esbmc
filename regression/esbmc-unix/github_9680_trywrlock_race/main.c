/* y never holds 1, so the assertion holds in every schedule (#9680). */
#include <assert.h>
#include <pthread.h>
pthread_rwlock_t l;
int y;
void *t(void *a)
{
  assert(y != 1);
  return 0;
}
int main(void)
{
  pthread_rwlock_init(&l, 0);
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_rwlock_trywrlock(&l);
  y = 2;
  pthread_join(th, 0);
  return 0;
}
