#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;
int **pp;

// Why the typed walk consults no size. The element type of `arr` has no
// constant size, so measuring it throws -- the offset walk has to give up and
// drop the path, which is how a pointer held in a variable-length element used
// to become invisible. The typed walk needs no size, because it has no offset
// to place, so the target stays reachable. This is the one shape that exercises
// that route's reason for existing rather than its member loop.
void *writer(void *arg)
{
  (void)arg;
  **pp = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  int n = nondet_int();
  __ESBMC_assume(n >= 1 && n <= 2);
  int *arr[2][n];
  arr[0][0] = &g;
  arr[1][0] = &g;
  pp = &arr[0][0];
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
