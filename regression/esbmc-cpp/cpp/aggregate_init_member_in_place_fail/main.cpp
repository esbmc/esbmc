#include <cassert>

int ctors = 0, dtors = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++ctors;
  }
  ~M()
  {
    ++dtors;
  }
};

struct W
{
  M m;
};

int main()
{
  {
    W a{M(5)};
    M arr[2] = {M(6), M(7)};
  }
  assert(dtors == 6);
  return 0;
}
