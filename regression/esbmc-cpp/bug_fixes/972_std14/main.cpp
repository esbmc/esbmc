// return type deducing
#include <cassert>

auto execute()
{
  return 1;
}

int main()
{
  assert(execute() == 1);
  return 0;
}
