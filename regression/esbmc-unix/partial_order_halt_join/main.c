/* #10990: main's assumption, as in reach_error under
 * --enable-unreachability-intrinsic, stops only main. A join on a thread that
 * ends still returns, so the claim before the assumption is checked. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  x = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  if (x != 1)
  {
    assert(0);
    __ESBMC_assume(0);
  }
  return 0;
}
