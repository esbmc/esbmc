#include <pthread.h>

void *malloc(unsigned size);
void free(void *p);

pthread_cond_t *cv;

void *thread1(void *arg)
{

  return NULL;
}


void *thread2(void *arg)
{

  return NULL;
}


int main(void)
{
  pthread_t id1, id2;

  cv = (pthread_cond_t *) malloc(sizeof(pthread_cond_t));

  pthread_cond_init(cv, NULL);

  pthread_create(&id1, NULL, thread1, NULL);
  pthread_create(&id2, NULL, thread2, NULL);

  pthread_join(id1, NULL);
  pthread_join(id2, NULL);

  free(cv);
  free(cv);

  return 0;
}
