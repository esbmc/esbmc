// C11 _Atomic with pthreads: data race on unprotected atomic variable
#include <pthread.h>

_Atomic int counter = 0;

void *inc(void *arg)
{
  counter = counter + 1;
  return 0;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, inc, 0);
  pthread_create(&t2, 0, inc, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
