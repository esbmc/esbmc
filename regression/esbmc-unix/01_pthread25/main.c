#include <pthread.h>

int g;

void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

void *t1(void *arg)
{
  // this is atomic!  
  __ESBMC_atomic_begin();
  g=2;
  g=1;
  __ESBMC_atomic_end();
}

int main()
{
  g=1;

  pthread_t id1;

  pthread_create(&id1, NULL, t1, NULL);
    
  assert(g==1);
}

