/* #10990: t2's write may come before t1's read. */
#include <pthread.h>
#include <assert.h>
int x;
void *t1(void *arg)
{
  int r = x;
  x = 1;
  assert(r == 0);
  return 0;
}
void *t2(void *arg)
{
  x = 2;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  return 0;
}
