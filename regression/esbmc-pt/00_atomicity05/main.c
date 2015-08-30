#include <pthread.h>

struct
{
  int a, b;
} s;

void *t1(void *arg)
{
  s.a=10; // not a race
}

void *t2(void *arg)
{
  s.b=20; // not a race
}

int main()
{
  pthread_t id1, id2;
  
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
}
