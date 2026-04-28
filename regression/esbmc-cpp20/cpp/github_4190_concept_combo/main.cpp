// github.com/esbmc/esbmc/issues/4190 — concept-style snippet from the issue.
// Verifies a `requires(std::is_enum_v<T>)` clause compiles + verifies under
// the patched bundled type_traits.

#include <type_traits>

template <typename T>
requires(std::is_enum_v<T>)
constexpr int kind() { return 1; }

enum class E { A, B, C };

int main()
{
  return kind<E>() == 1 ? 0 : 1;
}
