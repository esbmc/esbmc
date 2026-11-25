/* #10990: the index of an element write reads g, which t2 may change first.
   Without bounds checks, the index is the only read of g. */
#include <pthread.h>
#include <assert.h>
int a[2];
int g;
void *t1(void *arg)
{
  a[g] = 1;
  return 0;
}
void *t2(void *arg)
{
  g = 1;
  return 0;
}
int main()
{
  pthread_t x, y;
  pthread_create(&x, 0, t1, 0);
  pthread_create(&y, 0, t2, 0);
  pthread_join(x, 0);
  pthread_join(y, 0);
  assert(a[0] == 1);
  return 0;
}
