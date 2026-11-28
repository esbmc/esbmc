/* #10990: t2 dereferences a global pointer that t1, spawned after it, may
 * already have pointed at another global, so it reads x or y, never z. */
#include <pthread.h>
#include <assert.h>
int x, y = 1, z = 2;
int *p = &x;
void *t1(void *arg)
{
  p = &y;
  return 0;
}
void *t2(void *arg)
{
  int v = *p;
  assert(v != 2);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&b, 0, t2, 0);
  pthread_create(&a, 0, t1, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
