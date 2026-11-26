#include <pthread.h>
#include <assert.h>

__thread int data = 0;

void *worker(void *arg)
{
  assert(data == 5);
  return 0;
}

int main(void)
{
  pthread_t t;
  data = 5;
  pthread_create(&t, 0, worker, 0);
  pthread_join(t, 0);
  return 0;
}
