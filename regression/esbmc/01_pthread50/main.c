#include <errno.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>

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

  val = (long)pthread_getspecific(key);
  assert(val == 0);

  r = pthread_setspecific(key, nondet_bool() ? (void *)16 : NULL);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0 || r == EINVAL);

  return 0;
}
