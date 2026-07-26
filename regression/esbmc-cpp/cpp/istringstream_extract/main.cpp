// KNOWNBUG: extraction from an istringstream does not read its contents.
//
// std::istringstream's constructor discards the string it is given
// (src/cpp/library/sstream: the body is just `istream();`), and
// operator>>(istream&, int&) in src/cpp/library/istream never writes through
// its `val` argument -- it only resets _gcount. So `is >> v` silently leaves v
// exactly as it was, which for a stream built from a known string is a wrong
// answer rather than an under-approximation.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <sstream>
#include <string>
#include <cassert>

int main()
{
  // The insertion half of the model does work.
  std::ostringstream os;
  os << "ab";
  std::string s = os.str();
  assert(s.size() == 2);
  assert(s[0] == 'a');
  assert(s[1] == 'b');

  // The extraction half does not.
  std::istringstream is("42");
  int v = 0;
  is >> v;
  assert(v == 42);

  return 0;
}
