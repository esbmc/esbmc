// Exercises copy_memory_content's constant-size fast path:
// when both old and new element counts fold to a compile-time
// constant, the copy loop emits N unguarded assignments.
#include <assert.h>
#include <stdlib.h>

int main()
{
  int *a = (int *)malloc(4 * sizeof(int));
  if (!a)
    return 0;

  a[0] = 10;
  a[1] = 20;
  a[2] = 30;
  a[3] = 40;

  // Both 4 and 6 are constants → min(old=4, new=6) folds to 4.
  int *b = (int *)realloc(a, 6 * sizeof(int));
  if (!b)
  {
    // realloc failure leaves the original block untouched; free it.
    free(a);
    return 0;
  }

  // The first 4 elements must be preserved by realloc.
  assert(b[0] == 10);
  assert(b[1] == 20);
  assert(b[2] == 30);
  assert(b[3] == 40);

  free(b);
  return 0;
}
