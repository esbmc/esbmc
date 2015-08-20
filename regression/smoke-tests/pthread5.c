#include <pthread.h>

int x=0;

void *t1(void *arg)
{
  //this should fail
  assert(x==0 || x==1 || x==3);
}

void *t2(void *arg)
{
  x=1;
}

void *t3(void *arg)
{
  x=2;
}

int main()
{
  pthread_t id1, id2, id3;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  pthread_create(&id3, NULL, t3, NULL);

  return 0;
}
