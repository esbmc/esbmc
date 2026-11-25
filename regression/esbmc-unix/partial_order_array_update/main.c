/* #10990: an element write updates the array atomically, so writes to
   different elements from two threads are both kept. */
#include <pthread.h>
#include <assert.h>
int a[2];
void *t1(void *arg)
{
  a[0] = 1;
  return 0;
}
void *t2(void *arg)
{
  a[1] = 1;
  return 0;
}
int main()
{
  pthread_t x, y;
  pthread_create(&x, 0, t1, 0);
  pthread_create(&y, 0, t2, 0);
  pthread_join(x, 0);
  pthread_join(y, 0);
  assert(a[0] == 1 && a[1] == 1);
  return 0;
}
