#include <pthread.h>
#include <assert.h>

int g = 0;
int *a[2] = {0, &g};
int **ap = a + 1;

// R31, array element: the same erasure through the array arm of the descent --
// `a + 1` refers to the array symbol with the element index folded into a byte
// offset, which the value set keys as "a[]". Missed by the shape census that
// found R29 because it reached array elements only by indexing, never through
// a pointer into the array.
void *writer(void *arg)
{
  (void)arg;
  **ap = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
