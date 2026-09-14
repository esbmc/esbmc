#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_SOA_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_SOA_H_

#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <util/symtab/namespace.h>

class smt_tuple_soa_flattener;

/** Struct-of-arrays: an array of structs is stored as one native SMT array per
 *  scalar leaf, and an array nested inside a struct is hoisted out to its own
 *  array with the index linearised to `outer * extent + inner`.
 *
 *  What this buys over the node flattener is that every array in the formula
 *  is an array of scalars, so the solver's own array theory applies to all of
 *  it; the node flattener instead hands an array of structs to array_convt,
 *  which enumerates one variable per slot. It differs from the concat encoding
 *  in keeping a nested array as an array rather than burying it in a bitvector,
 *  where indexing it would need a variable shift. See #37.
 *
 *  Representation. `thetype` is always the logical ESBMC type of the value.
 *   - struct, or array whose element is a struct: `members` holds one child per
 *     struct member. A child of an array node has type "array of that member",
 *     so the decomposition pushes arrays inward through structs.
 *   - array whose flattened element is a scalar: `arr` is the fully flattened
 *     native array and `base` the element offset of this slice within it.
 *     `base` null means zero.
 *   - scalar: `arr` holds the value itself. */
class soa_ast : public smt_ast
{
public:
  soa_ast(
    smt_tuple_soa_flattener &_flat,
    smt_solver_baset *ctx,
    smt_sortt s,
    const type2tc &_thetype)
    : smt_ast(ctx, s), flat(_flat), thetype(_thetype)
  {
  }

  smt_astt
  ite(smt_solver_baset *ctx, smt_astt cond, smt_astt falseop) const override;
  smt_astt eq(smt_solver_baset *ctx, smt_astt other) const override;
  void assign(smt_solver_baset *ctx, smt_astt sym) const override;
  smt_astt update(
    smt_solver_baset *ctx,
    smt_astt value,
    unsigned int idx,
    const expr2tc &idx_expr) const override;
  smt_astt select(smt_solver_baset *ctx, const expr2tc &idx) const override;
  smt_astt project(smt_solver_baset *ctx, unsigned int elem) const override;

  void dump() const override;

  smt_tuple_soa_flattener &flat;
  type2tc thetype;

  std::vector<smt_astt> members;
  smt_astt arr = nullptr;
  smt_astt base = nullptr;
};

typedef const soa_ast *soa_astt;

inline soa_astt to_soa_ast(smt_astt a)
{
  soa_astt r = dynamic_cast<soa_astt>(a);
  assert(r != nullptr && "Non-SoA AST reached the SoA tuple flattener");
  return r;
}

class smt_tuple_soa_flattener : public tuple_iface
{
public:
  smt_tuple_soa_flattener(smt_solver_baset *_ctx, const namespacet &_ns)
    : ctx(_ctx), ns(_ns)
  {
  }

  ~smt_tuple_soa_flattener() override = default;

  smt_sortt mk_struct_sort(const type2tc &type) override;
  smt_astt tuple_create(const expr2tc &structdef) override;
  smt_astt tuple_fresh(smt_sortt s, std::string name = "") override;
  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override;
  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override;
  smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *inputargs,
    bool const_array,
    smt_sortt domain) override;
  smt_astt tuple_array_of(const expr2tc &init_value, unsigned long domain_width)
    override;
  expr2tc tuple_get(const expr2tc &expr) override;
  expr2tc tuple_get(const type2tc &type, smt_astt a) override;
  expr2tc tuple_get_array_elem(
    smt_astt array,
    uint64_t index,
    const type2tc &subtype) override;

  /** Build a value of @p type out of fresh symbols named after @p name. */
  smt_astt build(const std::string &name, const type2tc &type);

  /** Number of scalar slots a value of @p type occupies once its array
   *  dimensions are flattened; 1 for anything that is not an array. */
  uint64_t extent(const type2tc &type) const;

  /** Index sort used for the flattened leaf arrays of @p arrtype. */
  smt_sortt index_sort(const type2tc &arrtype) const;

  /** base + off, both widened to @p w -- which must be the *leaf* array's
   *  domain width. Deriving it from the logical type instead is wrong on a
   *  slice, whose type names only the inner dimension while its base counts
   *  in the flattened array. */
  smt_astt offset(smt_astt base, smt_astt off, std::size_t w) const;

  /** @p a resized to @p w bits. */
  smt_astt resize(smt_astt a, std::size_t w) const;

  smt_solver_baset *ctx;
  const namespacet &ns;
};

#endif /* SOLVERS_SMT_TUPLE_SMT_TUPLE_SOA_H_ */
