/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_Z3_CONV_H
#define CPROVER_PROP_Z3_CONV_H

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <solvers/prop/pointer_logic.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple.h>
#include <util/hash_cont.h>
#include <util/irep2.h>
#include <util/namespace.h>
#include <vector>
#include <z3pp.h>

class z3_smt_sort : public smt_sort {
public:
#define z3_sort_downcast(x) static_cast<const z3_smt_sort *>(x)
  z3_smt_sort(smt_sort_kind i, z3::sort _s)
    : smt_sort(i), s(_s), rangesort(nullptr) { }
  z3_smt_sort(smt_sort_kind i, z3::sort _s, const type2tc &_tupletype)
    : smt_sort(i), s(_s), rangesort(nullptr), tupletype(_tupletype) { }
  z3_smt_sort(smt_sort_kind i, z3::sort _s, unsigned long w)
    : smt_sort(i, w), s(_s), rangesort(nullptr) { }
  z3_smt_sort(smt_sort_kind i, z3::sort _s, unsigned long w, unsigned long dw,
              const smt_sort *_rangesort)
    : smt_sort(i, w, dw), s(_s), rangesort(_rangesort) { }

  ~z3_smt_sort() override = default;

  z3::sort s;
  const smt_sort *rangesort;
  type2tc tupletype;
};

class z3_smt_ast : public smt_ast {
public:
#define z3_smt_downcast(x) static_cast<const z3_smt_ast *>(x)
  z3_smt_ast(smt_convt *ctx, z3::expr _e, const smt_sort *_s) :
            smt_ast(ctx, _s), e(_e) { }
  ~z3_smt_ast() override = default;
  z3::expr e;

  const smt_ast *eq(smt_convt *ctx, const smt_ast *other) const override;
  const smt_ast *update(smt_convt *ctx, const smt_ast *value,
                                unsigned int idx, expr2tc idx_expr) const override;
  const smt_ast *select(smt_convt *ctx, const expr2tc &idx) const override;
  const smt_ast *project(smt_convt *ctx, unsigned int elem) const override;

  void dump() const override;
};

class z3_convt: public smt_convt, public tuple_iface, public array_iface
{
public:
  z3_convt(bool int_encoding, const namespacet &ns);
  ~z3_convt() override;
private:
  void intr_push_ctx();
  void intr_pop_ctx();
public:
  void push_ctx() override;
  void pop_ctx() override;
  smt_convt::resultt dec_solve() override;
  z3::check_result check2_z3_properties();

  expr2tc get_bool(const smt_ast *a) override;
  expr2tc get_bv(const type2tc &t, const smt_ast *a) override;
  expr2tc get_array_elem(const smt_ast *array, uint64_t index,
                                 const type2tc &subtype) override;

  void setup_pointer_sort();
  void convert_type(const type2tc &type, z3::sort &outtype);

  void convert_struct(const std::vector<expr2tc> &members,
                      const std::vector<type2tc> &member_types,
                      const type2tc &type, z3::expr &bv);

  void convert_struct_type(const std::vector<type2tc> &members,
                           const std::vector<irep_idt> &member_names,
                           const irep_idt &name, z3::sort &s);

  z3::expr mk_tuple_update(const z3::expr &t, unsigned i,
                           const z3::expr &new_val);
  z3::expr mk_tuple_select(const z3::expr &t, unsigned i);

  // SMT-abstraction migration:
  smt_astt mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs) override;
  smt_sort *mk_sort(const smt_sort_kind k, ...) override;

  smt_astt mk_smt_int(const mp_integer &theint, bool sign) override;
  smt_astt mk_smt_real(const std::string &str) override;
  smt_astt mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w) override;
  smt_astt mk_smt_bvfloat(const ieee_floatt &thereal,
                                  unsigned ew, unsigned sw) override;
  smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw) override;
  smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw) override;
  smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm) override;
  smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast) override;
  smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast) override;
  smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr) override;
  smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr) override;
  smt_astt mk_smt_bool(bool val) override;
  smt_astt mk_array_symbol(const std::string &name, const smt_sort *s,
                                   smt_sortt array_subtype) override;
  smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s) override;
  smt_sort *mk_struct_sort(const type2tc &type) override;
  smt_astt mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s) override;
  const smt_ast *make_disjunct(const ast_vec &v) override;
  const smt_ast *make_conjunct(const ast_vec &v) override;

  smt_astt tuple_create(const expr2tc &structdef) override;
  smt_astt tuple_fresh(const smt_sort *s, std::string name = "") override;
  expr2tc tuple_get(const expr2tc &expr) override;

  const smt_ast *tuple_array_create(const type2tc &array_type,
                                            const smt_ast **input_args,
                                            bool const_array,
                                            const smt_sort *domain) override;

  smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) override;
  smt_astt mk_tuple_array_symbol(const expr2tc &expr) override;
  smt_astt tuple_array_of(const expr2tc &init,
                                  unsigned long domain_width) override;

  const smt_ast *convert_array_of(smt_astt init_val,
                                          unsigned long domain_width) override;

  void add_array_constraints_for_solving() override;
  void add_tuple_constraints_for_solving() override;
  void push_array_ctx() override;
  void pop_array_ctx() override;
  void push_tuple_ctx() override;
  void pop_tuple_ctx() override;

  // Assert a formula; needs_literal indicates a new literal should be allocated
  // for this assertion (Z3_check_assumptions refuses to deal with assumptions
  // that are not "propositional variables or their negation". So we associate
  // the ast with a literal.
  void assert_formula(const z3::expr &ast);
  void assert_ast(const smt_ast *a) override;

  void debug_label_formula(const std::string&& name, const z3::expr &formula);
  void init_addr_space_array();

  const std::string solver_text() override
  {
    unsigned int major, minor, build, revision;
    Z3_get_version(&major, &minor, &build, &revision);
    std::stringstream ss;
    ss << "Z3 v" << major << "." << minor << "." << build;
    return ss.str();
  }

  tvt l_get(const smt_ast *a) override;

  void dump_smt() override;

  // Some useful types
public:

  inline z3_smt_ast *
  new_ast(z3::expr _e, const smt_sort *_s) {
    return new z3_smt_ast(this, _e, _s);
  }

  //  Must be first member; that way it's the last to be destroyed.
  z3::context ctx;
  z3::solver solver;
  z3::model model;

  bool smtlib, assumpt_mode;
  std::string filename;

  std::list<z3::expr> assumpt;
  std::list<std::list<z3::expr>::iterator> assumpt_ctx_stack;

  // Array of obj ID -> address range tuples
  z3::sort addr_space_tuple_sort;
  z3::sort addr_space_arr_sort;
  z3::func_decl addr_space_tuple_decl;

  // Debug map, for naming pieces of AST and auto-numbering them
  std::map<std::string, unsigned> debug_label_map;

  z3::sort pointer_sort;
  z3::func_decl pointer_decl;
};

#endif
