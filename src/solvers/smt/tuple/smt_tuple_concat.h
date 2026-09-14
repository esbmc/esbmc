#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_CONCAT_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_CONCAT_H_

#include <solvers/smt/smt_solver.h>
#include <solvers/smt/tuple/smt_tuple.h>
#include <util/symtab/namespace.h>

class smt_tuple_concat_flattener;

/** A struct held as one bitvector, members at constant bit offsets.
 *
 *  The point of this encoding is what it does one level up: an array of
 *  structs becomes a native SMT array whose range is that bitvector, so the
 *  solver's array theory still applies. The node flattener instead hands such
 *  an array to array_convt, which expands it to one variable per slot and an
 *  N-way ite per symbolic access, emitting no array operations at all.
 *
 *  Unions already use exactly this representation (smt_solver.cpp, union_id in
 *  convert_sort); this extends it to structs. Suggested for structs in #37. */
class concat_smt_ast : public smt_ast
{
public:
  concat_smt_ast(
    smt_tuple_concat_flattener &_flat,
    smt_solver_baset *ctx,
    smt_sortt s,
    smt_astt _inner,
    const type2tc &_thetype)
    : smt_ast(ctx, s), flat(_flat), inner(_inner), thetype(_thetype)
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

  void dump() const override
  {
    inner->dump();
  }

  smt_tuple_concat_flattener &flat;

  /** The bitvector holding the struct, or the native array of such
   *  bitvectors when thetype is an array of structs. */
  smt_astt inner;

  /** Struct type, or array-of-struct type. Held here rather than read back
   *  from the sort: the array sort's range is the bitvector sort, which has
   *  forgotten which struct it came from. */
  type2tc thetype;
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

  /** Width in bits of the bitvector representing @p type. */
  std::size_t bv_width(const type2tc &type) const;

  /** The bitvector sort a struct of @p type is held in. */
  smt_sortt bv_sort(const type2tc &type) const;

  /** Reinterpret a raw bitvector as a value of @p type, undoing to_bv(). */
  smt_astt from_bv(smt_astt raw, const type2tc &type);

  /** The bitvector holding @p a, whose ESBMC type is @p type. */
  smt_astt to_bv(smt_astt a, const type2tc &type);

  /** Wrap a bitvector as a struct-typed AST. */
  smt_astt wrap(smt_astt raw, const type2tc &type);

  smt_solver_baset *ctx;
  const namespacet &ns;
};

#endif /* SOLVERS_SMT_TUPLE_SMT_TUPLE_CONCAT_H_ */
