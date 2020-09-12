#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void __ESBMC_yield();

int x=0, y=0, z=0;

void* thread1(void* arg)
{
  if (x==0) x=1;
  if (y==0) y=1;
  //__ESBMC_yield();
  z=1;
  if (x==2 && y==2) assert(0);
}

void* thread2(void* arg)
{
  if (x==1) x=2;
  if (y==1) y=2;
}

void* thread3(void* arg)
{
  if (x==1) x=3;
  if (y==1) y=3;
}

void main()
{
  pthread_t id1, id2, id3;

  pthread_create(&id1, NULL, &thread1, NULL);
  pthread_create(&id2, NULL, &thread2, NULL);
  pthread_create(&id3, NULL, &thread3, NULL);  
}
