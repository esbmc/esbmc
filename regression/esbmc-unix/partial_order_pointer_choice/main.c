/* #10990: p may reach x or y, so the write through it updates both; the
   object it does not reach keeps the other thread's write. */
#include <pthread.h>
#include <assert.h>
int c, x, y;
void *t1(void *arg)
{
  int *p = c ? &x : &y;
  *p = 1;
  return 0;
}
void *t2(void *arg)
{
  y = 2;
  return 0;
}
int main()
{
  pthread_t a, b;
  c = 1;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 1 && y == 2);
  return 0;
}
