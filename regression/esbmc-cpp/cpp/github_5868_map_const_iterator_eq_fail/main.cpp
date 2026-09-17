#include <map>
#include <cassert>

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
  const std::map<int, payload> &cm = m;
  // A key that is not present compares equal to end().
  assert(cm.find(9) != cm.end());
  return 0;
}
