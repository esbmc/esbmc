// std::error_condition was missing from the bundled <system_error> (github
// #5868). The point of the type is the cross-category comparison: a
// system-category code matches a generic condition because system_category
// maps its values onto errno ([syserr.errcat.objects]/4), while two
// conditions compare on category identity alone.
#include <system_error>
#include <cassert>
#include <cstring>

int main()
{
  std::error_condition c;
  assert(!c);
  assert(c.value() == 0);
  assert(c.category() == std::generic_category());

  std::error_condition d =
    std::make_error_condition(std::errc::invalid_argument);
  assert(d);
  assert(d.value() == 22);
  assert(std::strcmp(d.category().name(), "generic") == 0);

  std::error_condition e = std::errc::invalid_argument; // implicit errc
  assert(e == d);

  std::error_condition f(22, std::system_category());
  assert(f != d); // conditions compare by category identity

  // A code carrying the platform's numbering matches the portable condition.
  std::error_code g(22, std::system_category());
  assert(g == d);
  assert(d == g);

  std::error_code h(22, std::generic_category());
  assert(h == d);

  std::error_code i(5, std::generic_category());
  assert(i != d);

  assert(std::generic_category().default_error_condition(22) == d);
  assert(g.default_error_condition() == d); // system value maps onto generic

  d.clear();
  assert(!d);
  return 0;
}
