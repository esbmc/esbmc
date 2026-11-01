// Replaced by its contract, erase removes at most one entry: the size relation
// holds, but the entries left are not described, so erasing another key may
// lose key 1. The model body keeps it.
#include <unordered_map>
#include <cassert>

int nondet_int();

int main()
{
  std::unordered_map<int, int> m;
  m[1] = 2;
  int k = nondet_int();
  size_t r = m.erase(k);
  assert(m.size() == 1 - r);
  assert(k == 1 || m.count(1) == 1);
  return 0;
}
