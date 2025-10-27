#include <cassert>
#include <functional>
#include <climits>

int main()
{
  std::hash<unsigned int> hasher;
    
  std::size_t hash1 = hasher(42U);
  std::size_t hash2 = hasher(0U);
  std::size_t hash3 = hasher(UINT_MAX);
    
  assert(hash1 >= 0);
  assert(hash2 >= 0);
  assert(hash3 >= 0);
    
  return 0;
}

