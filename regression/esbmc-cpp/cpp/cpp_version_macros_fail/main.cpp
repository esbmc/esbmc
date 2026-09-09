// Each macro <version> claims is tied here to the feature actually working, so
// the header cannot promise something the operational models do not provide.
#include <version>
#include <cassert>

#include <optional>
#include <variant>
#include <any>
#include <string_view>
#include <span>
#include <bit>
#include <source_location>
#include <expected>

int main()
{
#ifndef __cpp_lib_optional
#  error "<version> must claim __cpp_lib_optional"
#endif
  std::optional<int> o = 5;
  assert(o.has_value() && *o == 5);

#ifndef __cpp_lib_variant
#  error "<version> must claim __cpp_lib_variant"
#endif
  std::variant<int, double> v = 7;
  assert(v.index() == 0);

#ifndef __cpp_lib_any
#  error "<version> must claim __cpp_lib_any"
#endif
  std::any a = 5;
  assert(a.has_value());

#ifndef __cpp_lib_string_view
#  error "<version> must claim __cpp_lib_string_view"
#endif
  std::string_view sv = "ab";
  assert(sv.size() == 2);

#ifndef __cpp_lib_span
#  error "<version> must claim __cpp_lib_span"
#endif
  int arr[3] = {1, 2, 3};
  std::span<int> sp(arr, 3);
  assert(sp.size() == 4);

#ifndef __cpp_lib_bit_cast
#  error "<version> must claim __cpp_lib_bit_cast"
#endif
  assert(std::bit_cast<int>(1.0f) != 0);

#ifndef __cpp_lib_source_location
#  error "<version> must claim __cpp_lib_source_location"
#endif
  auto loc = std::source_location::current();
  (void)loc;

#ifndef __cpp_lib_expected
#  error "<version> must claim __cpp_lib_expected"
#endif
  std::expected<int, int> e = 1;
  assert(e.has_value());

  return 0;
}
