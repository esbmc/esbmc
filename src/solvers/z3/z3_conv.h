/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_Z3_CONV_H
#define CPROVER_PROP_Z3_CONV_H

#include <irep2.h>
#include <namespace.h>

#include <stdint.h>

#include <map>
#include <set>
#include <hash_cont.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/prop/pointer_logic.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple.h>
#include <vector>
#include <string.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include "z3pp.h"

typedef unsigned int uint;

class z3_convt: public smt_convt, public tuple_iface, public array_iface
{
public:
  z3_convt(bool int_encoding, const namespacet &ns);
  virtual ~z3_convt();
private:
  void intr_push_ctx(void);
  void intr_pop_ctx(void);
public:
  virtual void push_ctx(void);
  virtual void pop_ctx(void);
  virtual smt_convt::resultt dec_solve(void);
  z3::check_result check2_z3_properties(void);

  virtual expr2tc get_bool(const smt_ast *a);
  virtual expr2tc get_bv(const type2tc &t, const smt_ast *a);
  virtual expr2tc get_array_elem(const smt_ast *array, uint64_t index,
                                 const type2tc &subtype);

private:
  void setup_pointer_sort(void);
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
  virtual smt_astt mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast * const *args,
                               unsigned int numargs);
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...);

  virtual smt_astt mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_astt mk_smt_real(const std::string &str);
  virtual smt_astt mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_astt mk_smt_bvfloat(const ieee_floatt &thereal,
                                  unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_nan(unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw);
  virtual smt_astt mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm);
  virtual smt_astt mk_smt_typecast_from_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_typecast_to_bvfloat(const typecast2t &cast);
  virtual smt_astt mk_smt_nearbyint_from_float(const nearbyint2t &expr);
  virtual smt_astt mk_smt_bvfloat_arith_ops(const expr2tc &expr);
  virtual smt_astt mk_smt_bool(bool val);
  virtual smt_astt mk_array_symbol(const std::string &name, const smt_sort *s,
                                   smt_sortt array_subtype);
  virtual smt_astt mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  virtual smt_astt mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);
  virtual const smt_ast *make_disjunct(const ast_vec &v);
  virtual const smt_ast *make_conjunct(const ast_vec &v);

  virtual smt_astt tuple_create(const expr2tc &structdef);
  virtual smt_astt tuple_fresh(const smt_sort *s, std::string name = "");
  virtual expr2tc tuple_get(const expr2tc &expr);

  virtual const smt_ast *tuple_array_create(const type2tc &array_type,
                                            const smt_ast **input_args,
                                            bool const_array,
                                            const smt_sort *domain);

  virtual smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s);
  virtual smt_astt mk_tuple_array_symbol(const expr2tc &expr);
  virtual smt_astt tuple_array_of(const expr2tc &init,
                                  unsigned long domain_width);

  virtual const smt_ast *convert_array_of(smt_astt init_val,
                                          unsigned long domain_width);

  virtual void add_array_constraints_for_solving();
  virtual void add_tuple_constraints_for_solving();
  void push_array_ctx(void);
  void pop_array_ctx(void);
  void push_tuple_ctx();
  void pop_tuple_ctx();

  // Assert a formula; needs_literal indicates a new literal should be allocated
  // for this assertion (Z3_check_assumptions refuses to deal with assumptions
  // that are not "propositional variables or their negation". So we associate
  // the ast with a literal.
  void assert_formula(const z3::expr &ast);
  virtual void assert_ast(const smt_ast *a);

  void debug_label_formula(std::string name, const z3::expr &formula);
  void init_addr_space_array(void);

  virtual const std::string solver_text()
  {
    unsigned int major, minor, build, revision;
    Z3_get_version(&major, &minor, &build, &revision);
    std::stringstream ss;
    ss << "Z3 v" << major << "." << minor << "." << build;
    return ss.str();
  }

  virtual tvt l_get(const smt_ast *a);

  // Some useful types
public:
  #define z3_smt_downcast(x) static_cast<const z3_smt_ast *>(x)
  class z3_smt_ast : public smt_ast {
  public:
    z3_smt_ast(smt_convt *ctx, z3::expr _e, const smt_sort *_s) :
              smt_ast(ctx, _s), e(_e) { }
    virtual ~z3_smt_ast() { }
    z3::expr e;

    virtual const smt_ast *eq(smt_convt *ctx, const smt_ast *other) const;
    virtual const smt_ast *update(smt_convt *ctx, const smt_ast *value,
                                  unsigned int idx, expr2tc idx_expr) const;
    virtual const smt_ast *select(smt_convt *ctx, const expr2tc &idx) const;
    virtual const smt_ast *project(smt_convt *ctx, unsigned int elem) const;

    virtual void dump() const override;
  };

  inline z3_smt_ast *
  new_ast(z3::expr _e, const smt_sort *_s) {
    return new z3_smt_ast(this, _e, _s);
  }

  class z3_smt_sort : public smt_sort {
  public:
  #define z3_sort_downcast(x) static_cast<const z3_smt_sort *>(x)
    z3_smt_sort(smt_sort_kind i, z3::sort _s)
      : smt_sort(i), s(_s), rangesort(NULL) { }
    z3_smt_sort(smt_sort_kind i, z3::sort _s, const type2tc &_tupletype)
      : smt_sort(i), s(_s), rangesort(NULL), tupletype(_tupletype) { }
    z3_smt_sort(smt_sort_kind i, z3::sort _s, unsigned long w)
      : smt_sort(i, w), s(_s), rangesort(NULL) { }
    z3_smt_sort(smt_sort_kind i, z3::sort _s, unsigned long w, unsigned long dw,
                const smt_sort *_rangesort)
      : smt_sort(i, w, dw), s(_s), rangesort(_rangesort) { }

    virtual ~z3_smt_sort() { }

    z3::sort s;
    const smt_sort *rangesort;
    type2tc tupletype;
  };

  //  Must be first member; that way it's the last to be destroyed.
  z3::context ctx;
  z3::solver solver;
  z3::model model;

  bool smtlib, assumpt_mode;
  std::string filename;

  std::list<z3::expr> assumpt;
  std::list<std::list<z3::expr>::iterator> assumpt_ctx_stack;

  // XXX - push-pop will break here.
  typedef std::map<std::string, z3::expr> renumber_mapt;
  renumber_mapt renumber_map;

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
