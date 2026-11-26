/* #10990: the operational model's thread counters are not events. */
#include <pthread.h>
#include <assert.h>
extern unsigned short int __ESBMC_num_threads_running;
void *t(void *a)
{
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(__ESBMC_num_threads_running <= 2);
  return 0;
}
