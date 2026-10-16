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
  catch (const std::bad_variant_access &)
  {
    thrown = true;
  }
  assert(thrown);

  thrown = false;
  try
  {
    (void)std::get<0>(v);
  }
  catch (const std::exception &)
  {
    thrown = true;
  }
  assert(thrown);

  const std::variant<int, double> &c = v;
  thrown = false;
  try
  {
    (void)std::get<int>(c);
  }
  catch (...)
  {
    thrown = true;
  }
  assert(thrown);

  assert(std::get<double>(v) == 1.5);
  assert(std::get<1>(c) == 1.5);
  return 0;
}
