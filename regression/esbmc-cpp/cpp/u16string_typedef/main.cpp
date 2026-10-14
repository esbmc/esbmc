// std::u16string and std::u32string exist from C++11 on.
#include <cassert>
#include <string>
int main() {
  std::u16string a;
  std::u32string b;
  assert(a.empty() && b.empty());
  return 0;
}
