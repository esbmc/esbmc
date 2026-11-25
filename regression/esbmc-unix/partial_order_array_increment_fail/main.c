/* #10990: another thread's update may fall between the read of a[1] and the
   update, so an unprotected increment can be lost. */
#include <pthread.h>
#include <assert.h>
int a[2];
void *t(void *arg)
{
  int r = a[1];
  a[1] = r + 1;
  return 0;
}
int main()
{
  pthread_t x, y;
  pthread_create(&x, 0, t, 0);
  pthread_create(&y, 0, t, 0);
  pthread_join(x, 0);
  pthread_join(y, 0);
  assert(a[1] == 2);
  return 0;
}
