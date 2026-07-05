#include <cassert>

extern "C" int multi_tu_first();
extern "C" int multi_tu_second();

int main()
{
  // Both functions come from separate C++ translation units merged via
  // mergeASTs. The second TU's body must survive the merge.
  assert(multi_tu_first() == 1);
  assert(multi_tu_second() == 2);
  return 0;
}
