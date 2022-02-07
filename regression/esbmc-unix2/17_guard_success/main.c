#include <pthread.h>

_Bool x;
void *func(void *arg)
{
  if(!x)
    assert(0);
}

int main()
{
  pthread_t id;
  x = nondet_bool();
  if(x)
    pthread_create(&id, NULL, func, NULL);
  return 0;
}
