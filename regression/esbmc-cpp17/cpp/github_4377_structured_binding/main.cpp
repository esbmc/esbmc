#include <cassert>

struct P
{
  int a;
  int b;
};

int main()
{
  // Struct destructure.
  P p{1, 2};
  auto [x, y] = p;
  assert(x == 1);
  assert(y == 2);

  // Array destructure.
  int arr[3] = {10, 20, 30};
  auto [u, v, w] = arr;
  assert(u == 10);
  assert(v == 20);
  assert(w == 30);

  return 0;
}
