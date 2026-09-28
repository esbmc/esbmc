#include <locale>
#include <cassert>

int main()
{
  std::locale loc;
  assert(std::toupper('a', loc) == 'a');
  return 0;
}
