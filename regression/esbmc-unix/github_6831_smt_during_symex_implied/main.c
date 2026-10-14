#include <pthread.h>
#include <assert.h>

int x = 0;

void *writer(void *arg)
{
  (void)arg;
  x = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  pthread_join(t, 0);
  assert(x == 1);
  return 0;
}
