/* #10990: with c unset, p reaches y, and t1's write may follow t2's. */
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
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 0 && y == 2);
  return 0;
}
