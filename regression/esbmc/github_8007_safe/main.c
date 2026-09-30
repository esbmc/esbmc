// #8007
#include <assert.h>
#include <pthread.h>

int g = 0;
int h = 0;

void *writer(void *arg)
{
  h = 1;
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  int a = g + 1;
  int b = g + 1;
  assert(a == b);
  return 0;
}
