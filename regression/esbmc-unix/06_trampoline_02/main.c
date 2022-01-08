#include <pthread.h>
#include <assert.h>

int x;

void *thread1()
{
  if(x)
    assert(0);
  return NULL;
}

int main()
{
  pthread_t t1;
  pthread_create(&t1, 0, thread1, 0);
  return 0;
}
