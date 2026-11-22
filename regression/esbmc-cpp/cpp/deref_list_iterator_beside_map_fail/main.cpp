#include <cassert>
#include <list>
#include <unordered_map>

struct Cache
{
  std::unordered_map<int, std::list<int>::iterator> index;
  std::list<int> order;
};

int main()
{
  Cache c;
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 4);
  for (unsigned i = 0; i < n; ++i)
    c.order.push_back(1);
  if (nondet_bool() && !c.order.empty())
    c.order.erase(c.order.begin());
  unsigned sum = 0;
  for (std::list<int>::iterator it = c.order.begin(); it != c.order.end(); ++it)
    sum += *it;
  assert(sum <= 3);
  return 0;
}
