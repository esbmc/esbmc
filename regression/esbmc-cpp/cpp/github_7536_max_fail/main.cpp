// std::max reached through <iostream> must carry real semantics, not merely
// resolve (#7536).
#include <cassert>
#include <iostream>

int main()
{
  assert(std::max(1, 2) == 1);
  return 0;
}
