/*
 * Extracted from https://github.com/sosy-lab/sv-benchmarks).
 * Contributed by Vladimír Štill (https://divine.fi.muni.cz).
 * Description: A test case for pthread TLS.
 */

#include <errno.h>
#include <pthread.h>
#include <assert.h>

int main() {
  pthread_key_t key;
  int r = pthread_key_create( nondet_bool() ? &key : NULL, NULL );
  assert( r == 0 || r == EAGAIN || r == ENOMEM);
  return 0;
}

