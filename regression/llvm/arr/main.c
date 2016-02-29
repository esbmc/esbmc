#include <assert.h>

int arr1[2];

struct X
{
  int a;
  int b;
} x = {1};

int main()
{
  assert(arr1[0] == 0);
  assert(arr1[1] == 0);

  assert(x.b == 0);

  int arr[2] = {1};
  assert(arr[0] == 1);
  assert(arr[1] == 0);

  struct X x1;
  assert(x1.a != 0);

  return 0;
}
