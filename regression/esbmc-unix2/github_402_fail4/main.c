#include <pthread.h>
#include <assert.h>

int a,b;
void __VERIFIER_atomic_acquire(void)
{
  __VERIFIER_assume(a == 0);
  a = 1;
}
void* c(void *arg)
{
  b = 1;
  __VERIFIER_atomic_acquire();
  b = 1;
  return NULL;
}
pthread_t d;
int main()
{
  pthread_create(&d, 0, c, 0);
  __VERIFIER_atomic_acquire();
  if(b)
    assert(0);
  return 0;
}
