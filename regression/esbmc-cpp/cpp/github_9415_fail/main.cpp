#include <cassert>
#include <string>

int main()
{
  assert(std::to_string(4294967296ULL) == std::to_string(0ULL));
  return 0;
}
