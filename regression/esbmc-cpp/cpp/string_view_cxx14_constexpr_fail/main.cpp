#include <string_view>

/* char_traits::length is constexpr only from C++17 (P0426R1), so this
 * initialiser is not a constant expression in C++14. libc++ gates it the same
 * way, as _LIBCPP_CONSTEXPR_SINCE_CXX17, and clang++ -std=c++14 rejects this. */
constexpr std::string_view PREFIX = "@base@";

int main()
{
  return (int)PREFIX.size();
}
