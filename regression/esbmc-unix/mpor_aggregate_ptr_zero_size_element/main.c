#include <pthread.h>
#include <assert.h>

int g = 0;
int *a[2][0];
int **ap;

// Pins the `esize > 0` guard in collect_offset_paths. A zero-length inner array
// gives the element type size 0, and the descent reaches it with a *nonzero*
// byte offset -- both are needed, since BigInt short-circuits `0 % 0` to 0 and
// only a nonzero numerator reaches the divide. Without the guard `offset %
// esize` aborts the process from big-int, so this test dies by producing no
// verdict at all rather than the wrong one. The bounds violation is the point
// only in that it is a verdict.
void *writer(void *arg)
{
  (void)arg;
  **ap = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  ap = (int **)((char *)&a[0] + 1);
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
