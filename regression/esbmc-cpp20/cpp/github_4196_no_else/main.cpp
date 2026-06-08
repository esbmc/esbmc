#include <cassert>

int run()
{
  int x = 0;
  if consteval
  {
    x = 42;
  }
  return x;
}

int main()
{
  assert(run() == 0);
  return 0;
}
