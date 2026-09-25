#include <vector>
#include <cassert>

struct P
{
  int x, y;
  P(int a, int b) : x(a), y(b) {}
};

int main()
{
  std::vector<int> v;
  v.emplace_back(5);
  v.emplace_back(6);
  assert(v.size() == 2 && v[0] == 5 && v[1] == 6);

  // returned reference aliases the new element
  int &r = v.emplace_back(7);
  r = 9;
  assert(v[2] == 9);

  // emplace_back of a multi-argument type
  std::vector<P> vp;
  vp.emplace_back(3, 4);
  assert(vp[0].x == 3 && vp[0].y == 4);

  // growth over several elements
  std::vector<int> g;
  for (int i = 0; i < 5; i++)
    g.emplace_back(i * i);
  assert(g.size() == 5 && g[4] == 16);

  // shrink_to_fit keeps size and elements
  std::vector<int> s(4, 1);
  s.resize(2);
  s.shrink_to_fit();
  assert(s.size() == 2 && s[0] == 1);

  return 0;
}
