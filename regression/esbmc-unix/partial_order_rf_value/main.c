/* #10990: a shared read observes the value of some write to x. */
#include <pthread.h>
#include <assert.h>
int x;
void *t1(void *arg)
{
  x = 1;
  return 0;
}
void *t2(void *arg)
{
  int r = x;
  assert(r == 0 || r == 1);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  return 0;
}
