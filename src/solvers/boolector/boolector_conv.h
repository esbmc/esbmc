#ifndef _ESBMC_SOLVERS_BOOLECTOR_BOOLECTOR_CONV_H_
#define _ESBMC_SOLVERS_BOOLECTOR_BOOLECTOR_CONV_H_

#include <cstdio>
#include <solvers/smt/smt_conv.h>
#include <util/irep2.h>
#include <util/namespace.h>

extern "C" {
#include <boolector.h>
}

class btor_smt_ast : public solver_smt_ast<BoolectorNode *>
{
public:
  using solver_smt_ast<BoolectorNode *>::solver_smt_ast;
  ~btor_smt_ast() override = default;

  void dump() const override;
};

class boolector_convt : public smt_convt, public array_iface, public fp_convt
{
public:
  boolector_convt(bool int_encoding, const namespacet &ns);
  ~boolector_convt() override;

  resultt dec_solve() override;
  const std::string solver_text() override;

  void assert_ast(const smt_ast *a) override;

  smt_ast *mk_func_app(
    const smt_sort *s,
    smt_func_kind k,
    const smt_ast *const *args,
    unsigned int numargs) override;

  smt_sortt mk_bool_sort() override;
  smt_sortt mk_bv_sort(const smt_sort_kind k, std::size_t width) override;
  smt_sortt mk_array_sort(smt_sortt domain, smt_sortt range) override;
  smt_sortt mk_bv_fp_sort(std::size_t width, std::size_t swidth) override;
  smt_sortt mk_fpbv_rm_sort() override;

  smt_ast *mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_ast *mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bv(smt_sortt s, const mp_integer &theint) override;
  smt_ast *mk_smt_bool(bool val) override;
  smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_ast *mk_array_symbol(
    const std::string &name,
    const smt_sort *s,
    smt_sortt array_subtype) override;
  smt_ast *mk_extract(
    const smt_ast *a,
    unsigned int high,
    unsigned int low,
    const smt_sort *s) override;

  const smt_ast *
  convert_array_of(smt_astt init_val, unsigned long domain_width) override;

  void add_array_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;

  expr2tc get_bool(const smt_ast *a) override;
  expr2tc get_bv(const type2tc &type, smt_astt a) override;
  expr2tc get_array_elem(
    const smt_ast *array,
    uint64_t index,
    const type2tc &subtype) override;

  const smt_ast *overflow_arith(const expr2tc &expr) override;

  inline btor_smt_ast *new_ast(const smt_sort *_s, BoolectorNode *_e)
  {
    return new btor_smt_ast(this, _s, _e);
  }

  typedef BoolectorNode *(
    *shift_func_ptr)(Btor *, BoolectorNode *, BoolectorNode *);
  smt_ast *fix_up_shift(
    shift_func_ptr fptr,
    const btor_smt_ast *op0,
    const btor_smt_ast *op1,
    smt_sortt res_sort);

  void dump_smt() override;
  void print_model() override;

  // Members
  Btor *btor;

  typedef hash_map_cont<std::string, smt_ast *> symtable_type;
  symtable_type symtable;
};

#endif /* _ESBMC_SOLVERS_BOOLECTOR_BOOLECTOR_CONV_H_ */
