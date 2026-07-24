#include <map>
#include <cassert>

int main()
{
  std::map<int, int> m;

  // find on an empty map must equal end() (github #6354: != was unreliable)
  assert(m.find(1) == m.end());
  assert(!(m.find(1) != m.end()));

  m[1] = 10;
  m[2] = 20;

  // present / absent lookups
  assert(m.find(1) != m.end() && m.find(1)->second == 10);
  assert(m.find(9) == m.end());
  assert(!(m.find(9) != m.end()));

  // == and != are consistent
  auto f = m.find(1), e = m.end();
  assert((f == e) == (!(f != e)));

  // forward range traversal terminates correctly
  int c = 0, s = 0;
  for (std::map<int, int>::iterator it = m.begin(); it != m.end(); ++it)
  {
    c++;
    s += it->second;
  }
  assert(c == 2 && s == 30);

  // multimap uses the same iterator shape
  std::multimap<int, int> mm;
  mm.insert(std::make_pair(1, 2));
  assert(mm.find(1) != mm.end() && mm.find(9) == mm.end());

  return 0;
}
