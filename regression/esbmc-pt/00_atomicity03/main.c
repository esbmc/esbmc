#include <pthread.h>

int g;
int *p;

void *t1(void *arg)
{
  int l;
  
  l=*p;
}

void *t2(void *arg)
{
  g=1;
}

int main()
{
  pthread_t id1, id2;
  
  p=&g;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
}
