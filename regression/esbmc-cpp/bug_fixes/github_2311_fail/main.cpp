#include <cassert>
#include <cmath>

int fc()
{
  int c = -9;
  assert(std::abs(c) != 9);

  return c;
}

int main()
{
  return fc();
}
