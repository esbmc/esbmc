#include <algorithm>
#include <cassert>

// [algorithm.syn] opens with #include <initializer_list>, so naming
// std::initializer_list after including <algorithm> alone must compile.
int main()
{
  std::initializer_list<int> l = {4, 1, 3};
  assert(l.size() == 2);
  assert(*std::min_element(l.begin(), l.end()) == 1);
  return 0;
}
