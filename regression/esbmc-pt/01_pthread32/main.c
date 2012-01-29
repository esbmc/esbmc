#include <pthread.h>
#include <assert.h>

int x1 = 1;
int x2 = 2;
int x3 = 1;

_Bool flag1=0, flag2=0, flag3=0;

void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

void *t1(void *arg)
{
  __ESBMC_atomic_begin();
    x1 = (x3+1)%4;
  flag1=1;
  __ESBMC_atomic_end();
}

void *t2(void *arg)
{
  __ESBMC_atomic_begin();
    x2 = x1;
  flag2=1;
  __ESBMC_atomic_end();
}

void *t3(void *arg)
{
  __ESBMC_atomic_begin();
    x3 = x2;
  flag3=1;
  __ESBMC_atomic_end();
}

void *t4(void *arg)
{
  __ESBMC_atomic_begin();
  if (flag1 && flag2 && flag3)
    assert(x1 == x2 && x2 == x3);
  __ESBMC_atomic_end();
}

int main()
{
  pthread_t id1, id2, id3, id4;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  pthread_create(&id3, NULL, t3, NULL);
  pthread_create(&id3, NULL, t4, NULL);

  return 0;
}

