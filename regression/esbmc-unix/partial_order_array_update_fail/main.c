/* #10990: without joining t2, main may read a[1] before t2 writes it. */
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
  assert(a[0] == 1 && a[1] == 1);
  return 0;
}
