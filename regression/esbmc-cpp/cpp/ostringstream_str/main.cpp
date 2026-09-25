#include <cassert>
#include <sstream>
#include <string>

int main()
{
  std::ostringstream empty;
  assert(empty.str().size() == 0);

  std::ostringstream n;
  n << 42;
  assert(n.str() == "42");

  std::ostringstream neg;
  neg << -7;
  assert(neg.str() == "-7");

  std::ostringstream t;
  t << "ab";
  assert(t.str() == "ab");

  std::ostringstream c;
  c << 'z';
  assert(c.str() == "z");

  std::ostringstream b;
  b << true;
  assert(b.str() == "1");

  std::ostringstream u;
  u << 7u;
  assert(u.str() == "7");

  std::ostringstream seeded("xy");
  assert(seeded.str() == "xy");

  std::ostringstream set;
  set.str("hello");
  assert(set.str() == "hello");

  return 0;
}
