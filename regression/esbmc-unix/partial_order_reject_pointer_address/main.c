#include <pthread.h>
int x;
int *p;
void *t(void *arg)
{
  int **q = &p;
  *q = &x;
  return 0;
}
int main()
{
  pthread_t a;
  pthread_create(&a, 0, t, 0);
  pthread_join(a, 0);
  return 0;
}
