#include <stdlib.h>
#include <assert.h>

// Regression for esbmc/esbmc#5337: sizeof(T) is carried as a first-class node
// (no #c_sizeof_type side channel). malloc(sizeof(struct T)) must allocate a
// struct T (so field accesses are in-bounds), sizeof must keep clang's value
// in arithmetic, and n * sizeof(T) must compute the right byte count.

struct T
{
  int a;
  long b;
  void *c;
};

int main(void)
{
  // sizeof keeps its authoritative value in arithmetic contexts.
  assert(sizeof(struct T) >= sizeof(int) + sizeof(long) + sizeof(void *));
  assert(sizeof(int[10]) == 10 * sizeof(int));

  // malloc(sizeof(T)) is typed as struct T: writing every field is in-bounds.
  struct T *p = malloc(sizeof(struct T));
  if (p)
  {
    p->a = 1;
    p->b = 2;
    p->c = p;
    assert(p->a == 1 && p->b == 2 && p->c == p);
    free(p);
  }

  // n * sizeof(T) array allocation.
  unsigned n = 4;
  int *arr = malloc(n * sizeof(int));
  if (arr)
  {
    arr[0] = 7;
    arr[n - 1] = 9;
    assert(arr[0] == 7 && arr[n - 1] == 9);
    free(arr);
  }

  return 0;
}
