#include <pthread.h>
#include <assert.h>

int x1 = 1;
int x3 = 1;

_Bool flag1=0;

void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

void *t1(void *arg)
{
  __ESBMC_atomic_begin();
    x1 = (x3+1)%4;
  flag1=1;
  __ESBMC_atomic_end();
}

int main()
{
  pthread_t id1;

  pthread_create(&id1, NULL, t1, NULL);

  __ESBMC_atomic_begin();
  if (flag1)
    assert(x1 != x3);
  __ESBMC_atomic_end();

  return 0;
}

