#include <pthread.h>
#include <assert.h>

int x;

void* taskA(void* arg)
{
  x++;
  return NULL;
}

void* taskB(void* arg)
{
  if(!x)
    assert(1);
  return NULL;
}

int main() {
  pthread_t idA, idB;

  pthread_create(&idA, NULL, taskA, NULL);
  pthread_create(&idB, NULL, taskB, NULL);

  return 0;
}
