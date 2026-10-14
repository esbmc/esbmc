// KNOWNBUG: neither half of the string-stream model round-trips its contents.
//
// Extraction: std::istringstream's constructor discards the string it is given
// (src/cpp/library/sstream: the body is just `istream();`), and
// operator>>(istream&, int&) in src/cpp/library/istream never writes through
// its `val` argument -- it only resets _gcount. So `is >> v` silently leaves v
// exactly as it was, which for a stream built from a known string is a wrong
// answer rather than an under-approximation.
//
// Insertion: `os << "ab"; os.str()` does not yield "ab" either -- checked with
// --multi-property, all three of the size and character assertions below fail.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <sstream>
#include <string>
#include <cassert>

int main()
{
  // Insertion: str() should observe what was written.
  std::ostringstream os;
  os << "ab";
  std::string s = os.str();
  assert(s.size() == 2);
  assert(s[0] == 'a');
  assert(s[1] == 'b');

  // Extraction: the stream should parse the string it was built from.
  std::istringstream is("42");
  int v = 0;
  is >> v;
  assert(v == 42);

  return 0;
}
