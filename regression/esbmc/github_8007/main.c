// #8007
#include <assert.h>
#include <pthread.h>

int g = 0;

void *writer(void *arg)
{
  g = 1;
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  int a = g + 1;
  int b = g + 1;
  assert(a == b); // fails if writer runs between the two reads
  return 0;
}
