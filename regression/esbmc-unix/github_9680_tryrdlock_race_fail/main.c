/* A successful try must not leave the rest of main atomic (#9680). */
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
  pthread_rwlock_tryrdlock(&l);
  y = 1;
  y = 2;
  pthread_join(th, 0);
  return 0;
}
