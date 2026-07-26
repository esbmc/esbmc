#include <assert.h>

int main()
{
  int arr[4];
  int *a = &arr[0];
  int *b = &arr[3];

  // Relational comparison between pointers into the same array is
  // well-defined (C11 6.5.8p5), so this must not trip the same-object check.
  assert(a < b);

  return 0;
}
