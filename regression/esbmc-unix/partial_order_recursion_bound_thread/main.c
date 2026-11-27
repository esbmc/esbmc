/* #10990: t1 stops at its recursion bound, so its later write never runs. */
#include <pthread.h>
#include <assert.h>
int x;
void spin(void)
{
  spin();
}
void *t1(void *arg)
{
  spin();
  x = 1;
  return 0;
}
void *t2(void *arg)
{
  assert(x == 0);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
}
