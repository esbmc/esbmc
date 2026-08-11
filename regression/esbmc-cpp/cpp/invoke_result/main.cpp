#include <cassert>
#include <type_traits>

static int takes_int(int);
static char takes_two(int, long);

struct Functor
{
  double operator()(int) const;
};

int main()
{
  // [meta.trans.other]: invoke_result is the type of the call expression.
  static_assert(std::is_same<std::invoke_result_t<decltype(takes_int), int>,
                             int>::value, "int");
  static_assert(std::is_same<std::invoke_result_t<decltype(takes_two), int, long>,
                             char>::value, "char");
  static_assert(std::is_same<std::invoke_result_t<Functor, int>,
                             double>::value, "double");

  auto lam = [](int x) { return x + 1; };
  static_assert(std::is_same<std::invoke_result_t<decltype(lam), int>,
                             int>::value, "lambda");

  // The struct form agrees with the alias.
  static_assert(std::is_same<std::invoke_result<Functor, int>::type,
                             double>::value, "struct form");

  // SFINAE-friendly: no member `type` when the call is ill-formed, so this
  // does not hard-error.
  static_assert(!std::is_invocable<decltype(takes_int), const char *>::value,
                "not invocable");

  int v = 1;
  assert(v == 1);
  return 0;
}
