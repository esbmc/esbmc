#include <pthread.h>

void *thread1(void *arg)
{
  assert(0);
}

int main()
{
  pthread_t t1;
  pthread_create(&t1, 0, thread1, 0);
  return 0;
}
