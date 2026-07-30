// Guarding a header must not withdraw it from a mode where it already worked.
// PR #6526 first gated <string_view> and friends at C++17, which broke
// regression/esbmc/read_spec_verify_1 on Windows: with no --std, ESBMC passes
// no -std= at all and clang picks a per-target default that is below C++17
// there, so the header expanded to nothing. The guards are at C++11 instead --
// the lowest mode each of these already parsed in (github #3387).
#include <string_view>
#include <optional>
#include <any>
#include <variant>
#include <filesystem>
#include <source_location>
#include <string>
#include <functional>
#include <cassert>

int main()
{
  std::string_view sv("abc");
  assert(sv.size() == 3);
  assert(sv[1] == 'b');

  std::string s = "xy";
  std::string_view sv2 = s; // <string>'s operator string_view()
  assert(sv2.size() == 2);

  std::optional<int> o(7);
  assert(o.has_value());
  assert(*o == 7);

  std::any a = 11;
  assert(std::any_cast<int>(a) == 11);

  std::variant<int, char> v(5);
  assert(std::get<int>(v) == 5);
  return 0;
}
