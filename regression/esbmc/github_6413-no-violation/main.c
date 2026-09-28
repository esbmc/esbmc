#include <pthread.h>
#include <semaphore.h>

sem_t worker_done;
int shared;

void *worker(void *arg)
{
  int idx;
  for (idx = 0; idx < 2; ++idx)
  {
    shared = idx;
    __ESBMC_assert(1, "worker: in loop");
  }
  __ESBMC_assert(1, "worker: after loop");
  sem_post(&worker_done);
  return NULL;
}

int main()
{
  pthread_t thrd;
  sem_init(&worker_done, 0, 0);
  pthread_create(&thrd, NULL, worker, NULL);
  sem_wait(&worker_done);
  __ESBMC_assert(1, "main: after join");
  sem_destroy(&worker_done);
}
