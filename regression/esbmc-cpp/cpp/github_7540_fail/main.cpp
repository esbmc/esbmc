// The namespace-scope types must carry a real value, not merely parse: 7 is
// not 0 (#7540).
#include <cassert>
#include <ios>

int main()
{
  std::streamoff offset = 7;
  assert(offset == 0);
  return 0;
}
