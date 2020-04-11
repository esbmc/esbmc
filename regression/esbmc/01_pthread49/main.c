#include <pthread.h>
#include <assert.h>

int main()
{
  pthread_key_t key;
  int r = pthread_key_create(&key, NULL);
  assert(r == 1); //r must be 0
  return 0;
}
