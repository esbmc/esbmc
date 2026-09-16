#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_SOA_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_SOA_H_

#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <util/symtab/namespace.h>

class smt_tuple_soa_flattener;

/** Struct-of-arrays: an array of structs is stored as one native SMT array per
 *  scalar leaf, and an array nested inside it is flattened into its leaf with
 *  the index linearised to `outer * extent + inner`.
 *
 *  What this buys over the node flattener is that every array in the formula
 *  is an array of scalars, so the solver's own array theory applies to all of
 *  it; the node flattener instead hands an array of structs to array_convt,
 *  which enumerates one variable per slot. See #37.
 *
 *  `thetype` is the logical ESBMC type of the value. A soa_ast is one of:
 *   - a struct: `members` holds one term per member;
 *   - a node, an array whose element is a struct: `members` holds one term per
 *     member, each an array of that member with the node's dimensions;
 *   - a leaf: a member of array type inside a node, whose dimensions no ESBMC
 *     type spells. `arr` is its flattened native array. A row taken out of it
 *     is materialised as a plain backend array, so a leaf never reaches code
 *     outside this flattener -- which calls array_api directly on array terms.
 *  Every other array, including a struct's array member, is the plain backend
 *  array convert_sort describes. */
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

  bool leaf() const
  {
    return arr != nullptr;
  }

  smt_tuple_soa_flattener &flat;
  type2tc thetype;
  std::vector<smt_astt> members;
  smt_astt arr = nullptr;
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

  /** A value of @p type from fresh symbols named after @p name. @p in_node
   *  says it is a member of a node, the only place a leaf is needed. */
  smt_astt build(const std::string &name, const type2tc &type, bool in_node);

  /** Members of @p type as the SMT layer sees them: pointers and function
   *  pointers are pointer_struct, as in convert_sort. */
  std::vector<type2tc> members_of(const type2tc &type) const;

  /** Number of scalar slots a value of @p type occupies once its array
   *  dimensions are flattened; 1 for anything that is not an array, 0 for an
   *  array without a constant size. */
  uint64_t extent(const type2tc &type) const;

  /** Sort of the flattened native array holding the array type @p type. */
  smt_sortt flat_sort(const type2tc &type) const;

  /** @p a resized to @p w bits. */
  smt_astt resize(smt_astt a, std::size_t w) const;

  /** The backend array of type @p rowtype holding the slots of @p arr from
   *  position @p start. */
  smt_astt row(smt_astt arr, smt_astt start, const type2tc &rowtype);

  /** Constrain every slot of @p node, an array of @p type, to hold @p value. */
  void fill_const(smt_astt node, smt_astt value, const type2tc &type);

  smt_solver_baset *ctx;
  const namespacet &ns;
};

#endif /* SOLVERS_SMT_TUPLE_SMT_TUPLE_SOA_H_ */
