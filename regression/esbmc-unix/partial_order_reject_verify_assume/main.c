#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  __ESBMC_assume(x == 0);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  x = 1;
  return 0;
}
