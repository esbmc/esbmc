#include <assert.h>
#include <pthread.h>

int n;

void *worker(void *arg)
{
  return 0;
}

int main(int argc, char *argv[])
{
  pthread_t id;
  pthread_create(&id, 0, worker, 0);
  pthread_join(id, 0);
  assert(n == 1);
  return 0;
}
