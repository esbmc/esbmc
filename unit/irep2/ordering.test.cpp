// H-B1: expr2t/type2t comparison must be a strict total order (via lt) whose
// equality (via cmp) is an equivalence relation, with lt and cmp consistent
// (lt==0 <=> ==). A non-total order silently corrupts every std::set / map /
// hash-cons keyed on irep2 nodes (invariant I5). We sweep the REAL containers'
// operator< / operator== over a diverse corpus — mixed kinds, widths, BigInt
// signs, structurally-equal-but-distinct duplicates, and the unequal-length
// arrays that once tripped a do_type_lt OOB — asserting every order law on all
// pairs and transitivity on all triples.

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>

namespace
{
expr2tc testing_array(unsigned count)
{
  type2tc subtype = get_uint_type(config.ansi_c.word_size);
  type2tc array_ty = array_type2tc(subtype, expr2tc(), true);
  std::vector<expr2tc> members;
  for (unsigned i = 0; i < count; ++i)
    members.push_back(gen_ulong(i));
  return constant_array2tc(array_ty, members);
}

int sign(int x)
{
  return (x > 0) - (x < 0);
}

// Assert the strict-total-order + equivalence laws over a corpus of real
// containers (type2tc or expr2tc). Comparisons go through the canonical
// container operator< / operator== used for std::set/map keys.
template <typename C>
void check_order_laws(const std::vector<C> &corpus)
{
  const size_t n = corpus.size();

  // Every ordered pair, including the diagonal (i == j). Binding a and b as
  // distinct names keeps the comparisons off literal self-comparison forms.
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
    {
      const C &a = corpus[i];
      const C &b = corpus[j];
      const bool lt = a < b;
      const bool gt = b < a;
      const bool eq = (a == b);

      // Trichotomy: exactly one of a<b, b<a, a==b holds (strict total order).
      // On the diagonal this pins reflexivity of == and irreflexivity of <;
      // holding at both (i,j) and (j,i) also forces symmetry of ==.
      REQUIRE((int)lt + (int)gt + (int)eq == 1);
      // == is exactly the incomparability of <.
      REQUIRE(eq == (!lt && !gt));

      if (a && b) // underlying lt is only defined on non-nil nodes
      {
        // lt==0 <=> == (cmp/lt consistency).
        REQUIRE((a->lt(*b) == 0) == eq);
        // lt is sign-antisymmetric.
        REQUIRE(sign(a->lt(*b)) == -sign(b->lt(*a)));
      }
    }

  // Transitivity of < and of == over every triple.
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      for (size_t k = 0; k < n; ++k)
      {
        if (corpus[i] < corpus[j] && corpus[j] < corpus[k])
          REQUIRE(corpus[i] < corpus[k]);
        if (corpus[i] == corpus[j] && corpus[j] == corpus[k])
          REQUIRE(corpus[i] == corpus[k]);
      }
}
} // namespace

TEST_CASE("type2t ordering is a strict total order (H-B1)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  std::vector<irep_idt> names{"f0", "f1"};
  std::vector<type2tc> members{get_int_type(32), get_int_type(8)};

  expr2tc sz2 = constant_int2tc(get_uint_type(32), BigInt(2));
  expr2tc sz5 = constant_int2tc(get_uint_type(32), BigInt(5));

  std::vector<type2tc> corpus{
    type2tc(), // nil
    get_bool_type(),
    get_uint_type(8),
    get_uint_type(16),
    get_uint_type(32),
    get_uint_type(64),
    get_int_type(8),
    get_int_type(32),
    // Structurally equal to element [4] but a fresh allocation (get_uint_type
    // returns a shared singleton), so the distinct-pointer cmp path is exercised.
    unsignedbv_type2tc(32),
    array_type2tc(get_uint_type(8), sz2, false),
    array_type2tc(get_uint_type(8), sz5, false), // unequal array size
    vector_type2tc(get_uint_type(16), sz2),
    pointer_type2tc(get_uint_type(8)),
    struct_type2tc(members, names, names, "s"),
  };

  check_order_laws(corpus);
}

TEST_CASE("expr2t ordering is a strict total order (H-B1)", "[core][irep2]")
{
  config.ansi_c.word_size = 32;

  type2tc u32 = get_uint_type(32);
  expr2tc c5 = constant_int2tc(u32, BigInt(5));
  expr2tc c7 = constant_int2tc(u32, BigInt(7));

  std::vector<expr2tc> corpus{
    expr2tc(), // nil
    gen_true_expr(),
    gen_false_expr(),
    constant_int2tc(u32, BigInt(5)),
    constant_int2tc(u32, BigInt(7)),
    constant_int2tc(u32, BigInt(5)), // equal to element [3], distinct pointer
    constant_int2tc(get_int_type(64), BigInt(-5)),
    constant_int2tc(get_uint_type(8), BigInt(5)), // same value, different width
    symbol2tc(u32, "x"),
    symbol2tc(u32, "y"),
    symbol2tc(u32, "x"), // equal to element [8], distinct pointer
    add2tc(u32, c5, c7),
    testing_array(2),
    testing_array(5), // unequal member count
  };

  check_order_laws(corpus);
}
