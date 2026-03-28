// C11 _Atomic with pthreads: mutex-protected access, no data race
#include <pthread.h>

_Atomic int counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void *inc(void *arg)
{
  pthread_mutex_lock(&lock);
  counter = counter + 1;
  pthread_mutex_unlock(&lock);
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
