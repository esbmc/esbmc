#include <list>

int main()
{
  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  int n = 0;
  __ESBMC_unroll(1);
  for (auto i = l.begin(); i != l.end(); ++i)
    ++n;
  return n;
}
