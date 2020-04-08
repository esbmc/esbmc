#include <pthread.h>
#include <assert.h>

int main() {
  pthread_key_t key;
  int r = pthread_key_create( &key, NULL );
  assert( r == 0);
  return 0;
}

