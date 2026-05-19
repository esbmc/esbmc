// stringstream::operator<<(long double) must not be silently dropped by
// the free overload in <ostream>; without a member overload, the call is
// ambiguous between every integer overload at parse time.
#include <sstream>
#include <cassert>
using namespace std;

int main()
{
  stringstream oss;
  long double d = 45.543L;
  oss << d;
  assert(oss.str() == "45.543");
  return 0;
}
