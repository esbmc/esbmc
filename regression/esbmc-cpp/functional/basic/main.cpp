#include <cassert>
#include <functional>  // For std::hash
#include <climits>     // For INT_MAX and INT_MIN

int main()
{
  std::hash<int> hasher;
  // Test with various integer values
  std::size_t hash1 = hasher(42);
  std::size_t hash2 = hasher(-17);
  std::size_t hash3 = hasher(0);
  std::size_t hash4 = hasher(INT_MAX);
  std::size_t hash5 = hasher(INT_MIN);
  // Basic sanity checks - results should be valid size_t values
  assert(hash1 >= 0);  // Always true for size_t
  assert(hash2 >= 0);
  assert(hash3 >= 0);
  assert(hash4 >= 0);
  assert(hash5 >= 0);
  return 0;
}
