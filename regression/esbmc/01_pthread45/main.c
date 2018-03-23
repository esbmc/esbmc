#include <pthread.h>
#include <assert.h>

int x;

void * P0(void * arg)
{
  ++x;
  assert(x==2);
}

int main(void)
{
  pthread_t t1; 
  x=1;
  pthread_create(&t1, 0, P0, 0); 
  // pthread_join(t1, 0);
  assert(x==1);
  return 0;
}


