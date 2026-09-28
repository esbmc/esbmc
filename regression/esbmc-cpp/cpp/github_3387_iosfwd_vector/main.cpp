// <iosfwd> alongside another OM header used to resolve against the host
// libstdc++ copy, whose basic_ios<char> template clashed with the concrete
// std::ios the stream OMs define (github #3387).  -nostdinc++ plus a bundled
// <iosfwd> keeps the OM tree the sole source of these declarations.
#include <iosfwd>
#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.push_back(7);
  assert(v.size() == 1);
  assert(v[0] == 7);
  return 0;
}
