#include <pthread.h>

/* Each thread writes a different global object. Under --smt-during-symex
 * the address-space constraints must still keep A and B apart, otherwise
 * their data-race flags alias and a race is reported on A.datum. */
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
  int *d = &B.datum;
  pthread_create(&id, NULL, t_fun, NULL);
  pthread_mutex_lock(&B.mutex);
  *d = 8;
  pthread_mutex_unlock(&B.mutex);
  return 0;
}
