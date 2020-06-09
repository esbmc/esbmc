/*
 * Extracted from https://github.com/sosy-lab/sv-benchmarks.
 * Contributed by Vladimír Štill (https://divine.fi.muni.cz).
 * Description: A test case for pthread TLS. This test demonstartes that
 *              destructors are called.
*/

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>

void destructor(void *v)
{
  long val = (long)v;
  assert(val != 42); /* this assertion must fail */
}

void *worker(void *k)
{
  pthread_key_t *key = k;

  int r = pthread_setspecific(*key, (void *)42);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0 || r == EAGAIN);
  pthread_exit(NULL);
  return 0;
}

int main()
{
  pthread_key_t key;
  int r = pthread_key_create(&key, &destructor);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0 || r == EAGAIN);

  pthread_t tid;
  pthread_create(&tid, NULL, worker, &key);

  r = pthread_setspecific(key, (void *)16);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0 || r == EAGAIN);

  pthread_join(tid, NULL);
  return 0;
}
