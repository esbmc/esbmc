// Exercises <algorithm>'s heap helpers under -std=c++03. Those helpers used
// C++11 auto/decltype (and the bundled STL headers used C++11 '>>' template
// syntax), which failed to parse under C++03 until they were made portable.
#include <algorithm>
#include <cassert>

int main()
{
  int a[5] = {3, 1, 4, 1, 5};
  std::make_heap(a, a + 5);
  assert(a[0] == 5); // max at the root
  std::pop_heap(a, a + 5);
  assert(a[4] == 5); // max moved to the end
  return 0;
}
