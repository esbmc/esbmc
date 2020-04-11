/*
 * Extracted from https://github.com/sosy-lab/sv-benchmarks.
 * Contributed by Vladimír Štill (https://divine.fi.muni.cz).
 * Description: A test case for pthread TLS.
 */

#include <errno.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>

void *worker(void *k)
{
  pthread_key_t *key = k;

  long val = (long)pthread_getspecific(*key);
  assert(val == 0 || val == 16);

  int r = pthread_setspecific(*key, (void *)42);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0 || r == EINVAL);

  val = (long)pthread_getspecific(*key);
  assert(val == 16 || val == 42);

  return NULL;
}

int main()
{
  pthread_key_t key;
  int r = pthread_key_create(&key, NULL);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0 || r == EAGAIN);

  pthread_t tid;

  long val = (long)pthread_getspecific(key);
  assert(val == 0);

  pthread_create(&tid, NULL, worker, &key);

  val = (long)pthread_getspecific(key);
  assert(val == 0 || val == 42);

  r = pthread_setspecific(key, (void *)16);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0 || r == EINVAL);

  val = (long)pthread_getspecific(key);
  assert(val == 16 || val == 42);

  pthread_join(tid, NULL);
  val = (long)pthread_getspecific(key);
  assert(val == 16 || val == 42);

  return 0;
}
