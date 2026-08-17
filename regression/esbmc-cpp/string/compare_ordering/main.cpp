#include <cassert>
#include <string>

int main()
{
  std::string a = "a", b = "b";

  // [string.compare] is a three-way result: negative, zero or positive.
  assert(a.compare(b) < 0);
  assert(b.compare(a) > 0);
  assert(a.compare(a) == 0);

  // The common prefix decides before the lengths do: 'b' > 'a', so "b" > "aa"
  // even though "b" is the shorter string.
  std::string sh = "b", lo = "aa";
  assert(sh.compare(lo) > 0);
  assert(lo.compare(sh) < 0);

  // Equal prefix, different length: the shorter one is less.
  std::string p = "ab", q = "abc";
  assert(p.compare(q) < 0);
  assert(q.compare(p) > 0);

  assert(a.compare("b") < 0);
  assert(b.compare("a") > 0);

  std::string s = "abc", t = "abd";
  assert(s.compare(0, 2, "ab") == 0);
  assert(s.compare(0, 3, t, 0, 3) < 0);
  assert(t.compare(0, 3, s, 0, 3) > 0);

  assert(a < b);
  assert(b > a);
  assert(a <= b);

  return 0;
}
