#include <cassert>
#include <functional>

int main() 
{
  std::hash<bool> hasher;
    
  size_t hash_true = hasher(true);
  size_t hash_false = hasher(false);
    
  // Our model specifies that true->1, false->0
  assert(hash_true == 1);
  assert(hash_false == 0);
    
  // They should be different
  assert(hash_true != hash_false);
    
  return 0;
}

