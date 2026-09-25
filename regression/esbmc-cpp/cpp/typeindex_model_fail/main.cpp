// Negative counterpart of typeindex_model: distinct types really are
// distinguished, so a claim that they compare equal is refuted rather than
// vacuously held (issue #5868).
#include <typeindex>

int main()
{
  std::type_index i(typeid(int));
  std::type_index c(typeid(char));
  __ESBMC_assert(i == c, "must not hold");
  return 0;
}
