#include <cassert>

int main()
{
  assert([](int a) { return a + 2; }(2) == 444);
}