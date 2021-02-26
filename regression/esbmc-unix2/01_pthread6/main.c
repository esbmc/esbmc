#include <pthread.h>

int count;

void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

void *my_thread(void *arg)
{
  __ESBMC_atomic_begin();
  count++;
  count--;
  __ESBMC_atomic_end();
}

int main(void)
{
  pthread_t id;

  pthread_create(&id, NULL, my_thread, NULL);

  __ESBMC_atomic_begin();
  __ESBMC_assert(count==0, "property");
  __ESBMC_atomic_end();

  return 0;
}
