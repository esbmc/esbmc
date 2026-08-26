#include <locale>
#include <cassert>

int main()
{
  // [locale.convenience]: the two-argument forms boost's property_tree uses.
  std::locale loc;

  assert(std::toupper('a', loc) == 'A');
  assert(std::toupper('Z', loc) == 'Z');
  assert(std::tolower('Q', loc) == 'q');
  assert(std::tolower('7', loc) == '7');
  return 0;
}
