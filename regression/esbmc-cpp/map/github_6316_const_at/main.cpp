// github #6316: std::map::at must be callable on a const map. The model only
// declared the mutating overload, so at on a const map failed to compile. The
// const overload returns the mapped value and throws std::out_of_range when the
// key is absent, matching the standard and the vector::at model.
#include <cassert>
#include <map>
#include <stdexcept>

int lookup(const std::map<int, int> &m, int k)
{
  return m.at(k);
}

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[2] = 20;

  assert(lookup(m, 1) == 10);
  assert(lookup(m, 2) == 20);

  const std::map<int, int> &cm = m;
  bool threw = false;
  try { cm.at(99); }
  catch (std::out_of_range &) { threw = true; }
  assert(threw);
  return 0;
}
