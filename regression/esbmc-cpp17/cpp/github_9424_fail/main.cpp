#include <cassert>
#include <variant>

int main()
{
  std::variant<int, double> v(1.5);
  bool thrown = false;
  try
  {
    (void)std::get<int>(v);
  }
  catch (...)
  {
    thrown = true;
  }
  assert(!thrown);
  return 0;
}
