#include <pthread.h>

/* Both threads write A.datum, main without holding A.mutex: a real race. */
struct s
{
  int datum;
  pthread_mutex_t mutex;
} A, B;

void *t_fun(void *arg)
{
  pthread_mutex_lock(&A.mutex);
  A.datum = 5;
  pthread_mutex_unlock(&A.mutex);
  return NULL;
}

int main()
{
  pthread_t id;
  pthread_mutex_init(&A.mutex, NULL);
  pthread_mutex_init(&B.mutex, NULL);
  int *d = &A.datum;
  pthread_create(&id, NULL, t_fun, NULL);
  pthread_mutex_lock(&B.mutex);
  *d = 8;
  pthread_mutex_unlock(&B.mutex);
  return 0;
}
