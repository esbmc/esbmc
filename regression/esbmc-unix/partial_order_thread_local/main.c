#include <pthread.h>
#include <assert.h>

__thread int data = 0;

void *worker(void *arg)
{
  assert(data == 0);
  data = 1;
  assert(data == 1);
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  data = 5;
  pthread_create(&t1, 0, worker, 0);
  pthread_create(&t2, 0, worker, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  assert(data == 5);
  return 0;
}
