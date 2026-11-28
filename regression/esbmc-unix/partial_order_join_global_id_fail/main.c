#include <pthread.h>
#include <assert.h>
int x, y;
pthread_t g;
void *ta(void *arg)
{
  x = 1;
  return 0;
}
void *tb(void *arg)
{
  y = 1;
  return 0;
}
int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, ta, 0);
  pthread_create(&b, 0, tb, 0);
  g = a;
  pthread_join(g, 0);
  assert(y == 1);
  return 0;
}
