#include <pthread.h>
#include <semaphore.h>

sem_t sem;

int data;

void *producer(void *arg)
{
  data = 99;
  sem_post(&sem);
}

void *consumer(void *arg)
{
  sem_wait(&sem);
  __ESBMC_assert(data == 99, "assert data equal");
}

int main()
{
  sem_init(&sem, 0, 0);
  pthread_t prod;
  pthread_t cons;
  pthread_create(&prod, NULL, producer, NULL);
  pthread_create(&cons, NULL, consumer, NULL);
}
