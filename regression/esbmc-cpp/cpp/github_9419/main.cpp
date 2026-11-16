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
  assert(thrown);

  thrown = false;
  char buf[4];
  try
  {
    (void)v.copy(buf, 2, 4);
  }
  catch (std::out_of_range &)
  {
    thrown = true;
  }
  assert(thrown);

  std::string_view end = v.substr(3);
  assert(end.size() == 0);
  assert(v.substr(1, 5) == "bc");
  assert(v.copy(buf, 2, 1) == 2 && buf[0] == 'b' && buf[1] == 'c');
  return 0;
}
