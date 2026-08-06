// Negative counterpart of monostate_model: the variant really tracks which
// alternative it holds, so a claim contradicting the discriminator is refuted
// rather than vacuously held (issue #5868).
#include <variant>

int main()
{
  std::variant<std::monostate, int> v;
  v = 3;
  __ESBMC_assert(v.index() == 0, "must not hold");
  return 0;
}
