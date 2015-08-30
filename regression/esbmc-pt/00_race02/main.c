#include <pthread.h>

int g;
int *p;

//_Bool g_write_flag;

void *t1(void *arg)
{
  int l;
  
//  assert(g_write_flag==0);
  l=*p; // this is a R/W race
}

void *t2(void *arg)
{
//  assert(g_write_flag==0);
  g=1;
//  g_write_flag=1;
}

int main()
{
  pthread_t id1, id2;
  
  p=&g;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
}
