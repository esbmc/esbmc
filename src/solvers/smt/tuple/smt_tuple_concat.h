#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_CONCAT_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_CONCAT_H_

#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <unordered_map>
#include <util/symtab/namespace.h>

class smt_tuple_concat_flattener;

/** A struct, or an array of structs, under the concat encoding.
 *
 *  Only array elements are concatenated. An array of structs becomes a native
 *  SMT array whose range is one bitvector per element, which keeps the solver's
 *  array theory in play where the node flattener hands the array to
 *  array_convt and enumerates every slot (#37). A struct that is not an array
 *  element gains nothing from being one wide word -- every field write would
 *  rebuild the word -- so it is kept as one term per member.
 *
 *  Exactly one representation is used:
 *   - `inner` set: a packed struct, i.e. an array element, or an array of them.
 *     The layout is the SMT representation's, not C's: members are contiguous
 *     from bit 0, a pointer member takes pointer_struct's width, a bool one
 *     bit.
 *   - `inner` null: an unpacked struct, `members` holding one term per member.
 */
class concat_smt_ast : public smt_ast
{
public:
  concat_smt_ast(
    smt_tuple_concat_flattener &_flat,
    smt_solver_baset *ctx,
    smt_sortt s,
    const type2tc &_thetype,
    smt_astt _inner)
    : smt_ast(ctx, s), flat(_flat), thetype(_thetype), inner(_inner)
  {
  }

  concat_smt_ast(
    smt_tuple_concat_flattener &_flat,
    smt_solver_baset *ctx,
    smt_sortt s,
    const type2tc &_thetype,
    std::vector<smt_astt> _members)
    : smt_ast(ctx, s),
      flat(_flat),
      thetype(_thetype),
      members(std::move(_members))
  {
  }

  smt_astt
  ite(smt_solver_baset *ctx, smt_astt cond, smt_astt falseop) const override;
  smt_astt eq(smt_solver_baset *ctx, smt_astt other) const override;
  smt_astt update(
    smt_solver_baset *ctx,
    smt_astt value,
    unsigned int idx,
    const expr2tc &idx_expr) const override;
  smt_astt select(smt_solver_baset *ctx, const expr2tc &idx) const override;
  smt_astt project(smt_solver_baset *ctx, unsigned int elem) const override;
  void dump() const override;

  bool packed() const
  {
    return inner != nullptr;
  }

  smt_tuple_concat_flattener &flat;
  type2tc thetype;
  smt_astt inner = nullptr;
  std::vector<smt_astt> members;
};

typedef const concat_smt_ast *concat_smt_astt;

inline concat_smt_astt to_concat_ast(smt_astt a)
{
  concat_smt_astt ca = dynamic_cast<concat_smt_astt>(a);
  assert(ca != nullptr && "Non-concat AST reached the concat tuple flattener");
  return ca;
}

class smt_tuple_concat_flattener : public tuple_iface
{
public:
  smt_tuple_concat_flattener(smt_solver_baset *_ctx, const namespacet &_ns)
    : ctx(_ctx), ns(_ns)
  {
  }

  ~smt_tuple_concat_flattener() override = default;

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

  /** Members of @p type as the SMT layer sees them: pointers and function
   *  pointers are pointer_struct. */
  std::vector<type2tc> members_of(const type2tc &type) const;

  /** Bits @p type occupies in the packed layout; may be zero. */
  std::size_t width(const type2tc &type);

  /** Bit offset of member @p idx of @p type in the packed layout. */
  std::size_t offset(const type2tc &type, unsigned idx);

  /** Width of the bitvector a packed @p type is held in: never zero, since
   *  SMT has no zero-width sort. */
  std::size_t packed_width(const type2tc &type);

  /** @p a, a term of ESBMC type @p type, as packed_width(type) bits. */
  smt_astt to_bv(smt_astt a, const type2tc &type);

  /** The term of ESBMC type @p type whose packed bits are @p raw. */
  smt_astt from_bv(smt_astt raw, const type2tc &type);

  /** A value of @p type from fresh symbols named after @p name: unpacked for a
   *  struct, packed for an array of structs. */
  smt_astt build(const std::string &name, const type2tc &type);

  smt_solver_baset *ctx;
  const namespacet &ns;

private:
  /** Linux-driver structs nest deeply and every member access asks for an
   *  offset, so widths are computed once per type. */
  std::unordered_map<type2tc, std::size_t, type2_hash> width_cache;
};

#endif /* SOLVERS_SMT_TUPLE_SMT_TUPLE_CONCAT_H_ */
