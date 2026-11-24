/* #10990: t2's write of 1 may land between t1's last write and its read. */
#include <pthread.h>
#include <assert.h>
int x;
void *t1(void *arg)
{
  x = 1;
  x = 2;
  int r = x;
  assert(r != 1);
  return 0;
}
void *t2(void *arg)
{
  x = 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  return 0;
}
