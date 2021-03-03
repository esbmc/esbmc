#include <stdlib.h>
#include <assert.h>

typedef struct pthread_id_value
{
  int id;
  void *value;
} pthread_id_valuet;

/* This is an array with one element. */
pthread_id_valuet *__ESBMC_thread_keys[];

int main()
{
  pthread_id_valuet *id_val =
    (pthread_id_valuet *)malloc(sizeof(pthread_id_valuet));
  __ESBMC_thread_keys[0] =
    (pthread_id_valuet *)malloc(sizeof(pthread_id_valuet));

  id_val->id = 10;
  id_val->value = (void *)20;
  __ESBMC_thread_keys[0][0] = *id_val;

  /* array bounds violated: array `__ESBMC_thread_keys' upper bound */
  __ESBMC_thread_keys[1] =
    (pthread_id_valuet *)malloc(sizeof(pthread_id_valuet));
  id_val->id = 40;
  id_val->value = (void *)50;
  __ESBMC_thread_keys[1][0] = *id_val;

  pthread_id_valuet *tmp = &__ESBMC_thread_keys[0][0];
  assert(tmp->id == 10);
  assert(tmp->value == (void *)20);

  tmp = &__ESBMC_thread_keys[1][0];
  assert(tmp->id == 40);
  assert(tmp->value == (void *)50);

  return 0;
}
