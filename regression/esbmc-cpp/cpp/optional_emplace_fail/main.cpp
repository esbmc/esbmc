#include <optional>
#include <cassert>

int main()
{
  std::optional<int> o;
  o.emplace(7);
  assert(*o == 99); // wrong value
  return 0;
}
