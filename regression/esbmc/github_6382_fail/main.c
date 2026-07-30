#include <pthread.h>
#include <semaphore.h>

sem_t sem;

int data;
int data_vfy;

void *prod_task(void *arg)
{
  data = data_vfy;
  sem_post(&sem);
  return NULL;
}

void *cons_task(void *arg)
{

  __ESBMC_assert(data == data_vfy, "assert data equal");
  return NULL;
}

int main()
{
  data_vfy = nondet_int();
  sem_init(&sem, 0, 0);

  pthread_t prod_thrd;
  pthread_t cons_thrd;

  pthread_create(&prod_thrd, NULL, prod_task, NULL);
  pthread_create(&cons_thrd, NULL, cons_task, NULL);

  pthread_join(prod_thrd, NULL);
  pthread_join(cons_thrd, NULL);

  sem_destroy(&sem);
}
