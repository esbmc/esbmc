// <chrono> includes <limits>, so it was collateral damage of the same
// pre-C++17 parse failure (github #3387).
#include <chrono>
#include <cassert>

int main()
{
  std::chrono::seconds s(3);
  std::chrono::milliseconds ms = std::chrono::duration_cast<
    std::chrono::milliseconds>(s);
  assert(ms.count() == 3000);
  return 0;
}
