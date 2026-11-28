/* #10990: a write through a global pointer that t1, spawned after t2, may
 * already have pointed at another global updates x or y. */
#include <pthread.h>
#include <assert.h>
int x, y;
int *p = &x;
void *t1(void *arg)
{
  p = &y;
  return 0;
}
void *t2(void *arg)
{
  *p = 5;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&b, 0, t2, 0);
  pthread_create(&a, 0, t1, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x + y == 5 && p == &y);
  return 0;
}
