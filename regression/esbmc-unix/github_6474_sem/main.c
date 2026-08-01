#include <pthread.h>
#include <semaphore.h>

static sem_t s;

static void *worker(void *p)
{
  (void)p;
  sem_wait(&s);
  sem_post(&s);
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  sem_init(&s, 0, 1);
  sem_wait(&s);
  pthread_create(&t1, 0, worker, 0);
  pthread_create(&t2, 0, worker, 0);
  sem_post(&s);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  sem_destroy(&s);
  return 0;
}
