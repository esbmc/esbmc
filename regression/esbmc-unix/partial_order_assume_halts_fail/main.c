/* #10990: a thread whose assumption fails stops there, so its later write
 * never happens. */
#include <pthread.h>
#include <assert.h>
int x, y;
void *t(void *arg)
{
  __ESBMC_assume(x == 1);
  y = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  x = 1;
  assert(y == 0);
  return 0;
}
