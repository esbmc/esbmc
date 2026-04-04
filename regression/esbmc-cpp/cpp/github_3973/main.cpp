#include <list>

int main()
{
  std::list<int> xs;
  int x;
  xs.push_front(x);
  __ESBMC_assert(xs.front() == x, "assert");
  return 0;
}
