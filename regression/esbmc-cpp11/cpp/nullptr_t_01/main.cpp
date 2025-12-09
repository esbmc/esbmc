#include <cassert>
#include <cstddef>

bool f(std::nullptr_t x)
{
  return x == &x;
}
int main()
{ 
  assert(!f(nullptr)); 
}
