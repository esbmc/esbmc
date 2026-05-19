// std::string iterator copies must propagate `it_str` (the underlying
// buffer pointer) so that range-based append/assign can read the source.
// Before the fix, `iterator(const iterator &)` copied only `pointer` and
// `pos`, so `s.begin()+k` returned an iterator with an uninitialised
// `it_str`, and `str.append(s.begin()+k, s.end())` read garbage.
#include <string>
#include <cassert>

int main()
{
  // Templated append<int>(n, c): tests the integer-iterator overload that
  // must NUL-terminate its scratch buffer (the `*this = temp` follow-up
  // strcpy's into a buffer whose terminator was previously not written).
  std::string a;
  a.append<int>(5, '*');
  assert(a == "*****");

  // Range append from a non-default-iterator (begin()+k, end()): exercises
  // the iterator copy-ctor it_str propagation.
  std::string b;
  std::string src = "hello world";
  b.append(src.begin() + 6, src.end());
  assert(b == "world");

  // Range assign from (begin()+k, end()-k): exercises both endpoints
  // moving away from the trivial begin()/end() positions.
  std::string c;
  c.assign(src.begin() + 6, src.end() - 1);
  assert(c == "worl");

  return 0;
}
