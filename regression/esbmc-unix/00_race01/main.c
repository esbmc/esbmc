#include <pthread.h>

int g;

void *t1(void *arg)
{
  int l;
  
  l=g; // this is a R/W race
}

void *t2(void *arg)
{
  g=1;
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
}
