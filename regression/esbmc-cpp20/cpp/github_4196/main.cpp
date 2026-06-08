#include <cassert>

int run()
{
  int x = 0;
  if consteval
  {
    x = 42;
  }
  else
  {
    x = 1;
  }
  return x;
}

int main()
{
  assert(run() == 1);
  return 0;
}
