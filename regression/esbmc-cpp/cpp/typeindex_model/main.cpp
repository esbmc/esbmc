// Model for <typeindex>. std::type_index wraps a `const type_info &` so a type
// can key an associative container; the ordering is required to be consistent
// within a run but not stable across runs ([type.index]), which is what the
// underlying type_info model already provides.
//
// Missing entirely before, and the largest single header blocking ESBMC's own
// translation units from parsing (issue #5868).
#include <typeindex>

int main()
{
  std::type_index i1(typeid(int));
  std::type_index i2(typeid(int));
  std::type_index c(typeid(char));

  __ESBMC_assert(i1 == i2, "the same type compares equal");
  __ESBMC_assert(!(i1 != i2), "and not unequal");
  __ESBMC_assert(i1 != c, "different types compare unequal");

  // A strict weak ordering: irreflexive, and antisymmetric across a pair.
  __ESBMC_assert(!(i1 < i1), "irreflexive");
  __ESBMC_assert((i1 < c) != (c < i1), "antisymmetric");

  // The relational operators agree with each other.
  __ESBMC_assert((i1 < c) == (c > i1), "< and > agree");
  __ESBMC_assert((i1 <= i2) && (i1 >= i2), "equal compares both <= and >=");

  // hash is consistent with equality, which is what container use relies on.
  std::hash<std::type_index> h;
  __ESBMC_assert(h(i1) == h(i2), "equal keys hash equal");
  __ESBMC_assert(i1.hash_code() == i2.hash_code(), "hash_code agrees");

  __ESBMC_assert(i1.name() == i2.name(), "name is the shared type-name string");
  return 0;
}
