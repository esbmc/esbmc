#include <pthread.h>
#include <assert.h>

void *thread1()
{
  assert(0);
  return NULL;
}

int main()
{
  pthread_t t1;
  pthread_create(&t1, 0, thread1, 0);
  return 0;
}
