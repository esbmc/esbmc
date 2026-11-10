// #10171: an aligned array member keeps its initializer.
#include <assert.h>
struct S
{
  int a[4] __attribute__((aligned(16)));
  int b;
};
struct S s = {{1, 2, 3, 4}, 5};
int main()
{
  assert(s.a[0] == 1 && s.a[3] == 4 && s.b == 5);
  return 0;
}
