#include <map>
#include <cassert>

// A mapped type with no operator== -- exactly what goto_functiont is.
struct payload
{
  int v;
  payload() : v(0)
  {
  }
  payload(int x) : v(x)
  {
  }
};

int main()
{
  std::map<int, payload> m;
  m[1] = payload(10);

  // Comparing const_iterators must not require mapped_type to be comparable.
  const std::map<int, payload> &cm = m;
  assert(cm.find(1) != cm.end());
  assert(cm.find(9) == cm.end());
  assert(cm.begin() != cm.end());
  return 0;
}
