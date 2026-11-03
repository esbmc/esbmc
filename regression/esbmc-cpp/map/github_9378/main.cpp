// #9378: map::erase(first, last) erases [first, last), also when first is
// not begin().
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  for (int i = 0; i < 5; i++)
    m[i] = i;
  std::map<int, int>::iterator f = m.begin();
  ++f;
  std::map<int, int>::iterator l = f;
  ++l;
  ++l;
  m.erase(f, l);
  assert(m.size() == 3);
  assert(m.count(0) == 1 && m.count(1) == 0 && m.count(2) == 0);
  assert(m.count(3) == 1 && m.count(4) == 1);
  assert(m[3] == 3 && m[4] == 4);

  std::multimap<int, int> mm;
  mm.insert(std::make_pair(1, 10));
  mm.insert(std::make_pair(2, 20));
  mm.insert(std::make_pair(2, 21));
  mm.insert(std::make_pair(3, 30));
  std::multimap<int, int>::iterator g = mm.begin();
  ++g;
  std::multimap<int, int>::iterator h = g;
  ++h;
  ++h;
  mm.erase(g, h);
  assert(mm.size() == 2 && mm.count(2) == 0);
  assert(mm.count(1) == 1 && mm.count(3) == 1);
  return 0;
}
