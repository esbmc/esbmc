/* #10990: a thread whose assumption fails stops there; main's join on
 * another thread, which ends, still returns. */
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
  __ESBMC_assume(x == 2);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  assert(x != 1);
  return 0;
}
