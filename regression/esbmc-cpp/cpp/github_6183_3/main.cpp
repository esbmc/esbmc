// <cstdlib> alone must make std::size_t visible, as libstdc++ does.
#include <cstdlib>

int main()
{
  std::size_t n = 3;
  return n == 3 ? 0 : 1;
}
