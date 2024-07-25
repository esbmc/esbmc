#include <pthread.h>
#include <assert.h>
#include <stdio.h>

int nondet_int();
int x=0, w=0, z=0;

void *t1(void *arg)
{
  if (w==0 && z==0)
    printf("x: %i\n", x);
  return NULL;
}

void *t2(void *arg)
{
  int y=nondet_int();
  if (y==0 || y==1) 
  {
    x=0;
    y=0;
  }
  return NULL;
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  return 0;
}
