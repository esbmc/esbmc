/*
 * Extracted from https://github.com/sosy-lab/sv-benchmarks.
 * Contributed by Vladimír Štill (https://divine.fi.muni.cz).
 * Description: A test case for pthread TLS.
 */

#include <pthread.h>
#include "svc.h"

void destructor(void *v)
{
  long val = (long)v;
  assert(val != 42); // triggered by destructor of workers's version of TLS
}

void *worker(void *k)
{
  pthread_key_t *key = k;

  int r = pthread_setspecific(*key, (void *)42);
  assert(r == 0);
  return 0;
}

int main()
{
  pthread_key_t key;
  int r = pthread_key_create(&key, &destructor);
  assert(r == 0);

  pthread_t tid;
  pthread_create(&tid, NULL, worker, &key);

  r = pthread_setspecific(key, (void *)16);
  assert(r == 0);

  pthread_join(tid, NULL);
}
