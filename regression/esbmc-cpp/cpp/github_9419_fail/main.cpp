#include <cassert>
#include <stdexcept>
#include <string_view>

int main()
{
  std::string_view v("abc");
  bool thrown = false;
  try
  {
    (void)v.substr(5);
  }
  catch (std::out_of_range &)
  {
    thrown = true;
  }
  assert(!thrown);
  return 0;
}
