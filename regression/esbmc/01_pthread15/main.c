#include <pthread.h>

int x;

void *t1(void *arg)
{
  x = 1;
  x = 2;
}

void *t2(void *arg)
{
  x = 3;
  x = 4;
  x = 5;
  assert(x==5);
}

int main()
{
  pthread_t id1, id2;
  
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);  
}
