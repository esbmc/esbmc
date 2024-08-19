#include <cassert>
typedef int a;
a b;
int main()
{
  b = 22;
  b.~a();
  assert(b == 2222);
}
