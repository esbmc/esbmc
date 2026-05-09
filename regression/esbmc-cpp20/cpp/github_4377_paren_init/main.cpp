#include <cassert>

struct S
{
  int a;
  int b;
  int c;
};

int main()
{
  S s(1, 2, 3);
  assert(s.a == 1);
  assert(s.b == 2);
  assert(s.c == 3);

  // Array variant.
  int arr[](10, 20, 30);
  assert(arr[0] == 10);
  assert(arr[2] == 30);
  return 0;
}
