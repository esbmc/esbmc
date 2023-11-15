#include <cassert>

auto execute()
{
  return 1;
}

int main()
{
  assert(execute() == 0); // should be 1
  return 0;
}
