/* #10990: nothing after __ESBMC_terminate_thread runs. */
#include <pthread.h>
#include <assert.h>
void __ESBMC_terminate_thread(void);
int x = 0;
void *t(void *a)
{
  x = 1;
  __ESBMC_terminate_thread();
  x = 2;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(x != 2);
  return 0;
}
