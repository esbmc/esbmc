#include <errno.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>

int main()
{
  pthread_key_t key;
  int r = pthread_key_create(nondet_bool() ? &key : NULL, NULL);
  if(r == ENOMEM)
  {
    exit(1);
  }
  assert(r == 0);
  return 0;
}
