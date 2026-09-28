// reserve() reallocates in place rather than allocating a second buffer and
// copying, so the elements already stored must survive every growth.
#include <vector>
#include <cassert>

struct P
{
  int a;
  int b;
};

int main()
{
  std::vector<int> v;
  v.push_back(7);
  v.push_back(8);
  v.reserve(64); // crosses the default capacity, so the body runs
  assert(v[0] == 7);
  assert(v[1] == 8);
  assert(v.size() == 2);

  for (int i = 0; i < 18; i++)
    v.push_back(i);
  assert(v.size() == 20);
  assert(v[0] == 7);
  assert(v[19] == 17);

  std::vector<P> s;
  for (int i = 0; i < 12; i++)
  {
    P p;
    p.a = i;
    p.b = i * 2;
    s.push_back(p);
  }
  assert(s[0].a == 0);
  assert(s[11].a == 11);
  assert(s[11].b == 22);

  return 0;
}
