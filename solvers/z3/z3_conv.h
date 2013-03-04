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
#include <solvers/prop/prop_conv.h>
#include <solvers/prop/pointer_logic.h>
#include <solvers/smt/smt_conv.h>
#include <vector>
#include <string.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include "z3++.h"

typedef unsigned int uint;

class z3_convt: public smt_convt
{
public:
  z3_convt(bool int_encoding, bool is_cpp, const namespacet &ns);
  virtual ~z3_convt();
private:
  void intr_push_ctx(void);
  void intr_pop_ctx(void);
public:
  virtual void push_ctx(void);
  virtual void pop_ctx(void);
  virtual prop_convt::resultt dec_solve(void);
  z3::check_result check2_z3_properties(void);

  // overloading
  virtual expr2tc get(const expr2tc &expr);

private:
  bool assign_z3_expr(const exprt expr);
  u_int convert_member_name(const exprt &lhs, const exprt &rhs);

  void setup_pointer_sort(void);
  void convert_type(const type2tc &type, z3::sort &outtype);

  void convert_struct_union(const std::vector<expr2tc> &members,
                            const std::vector<type2tc> &member_types,
                            const type2tc &type, z3::expr &bv);

  void convert_struct_union_type(const std::vector<type2tc> &members,
                                 const std::vector<irep_idt> &member_names,
                                 const irep_idt &name, bool uni, void *_bv);

  z3::expr mk_tuple_update(const z3::expr &t, unsigned i,
                           const z3::expr &new_val);
  z3::expr mk_tuple_select(const z3::expr &t, unsigned i);

  // SMT-abstraction migration:
  virtual smt_ast *mk_func_app(const smt_sort *s, smt_func_kind k,
                               const smt_ast **args, unsigned int numargs);
  virtual smt_sort *mk_sort(const smt_sort_kind k, ...);
  virtual literalt mk_lit(const smt_ast *s);

  virtual smt_ast *mk_smt_int(const mp_integer &theint, bool sign);
  virtual smt_ast *mk_smt_real(const mp_integer &theint);
  virtual smt_ast *mk_smt_bvint(const mp_integer &theint, bool sign,
                                unsigned int w);
  virtual smt_ast *mk_smt_bool(bool val);
  virtual smt_ast *mk_smt_symbol(const std::string &name, const smt_sort *s);
  virtual smt_sort *mk_struct_sort(const type2tc &type);
  virtual smt_sort *mk_union_sort(const type2tc &type);
  virtual smt_ast *mk_extract(const smt_ast *a, unsigned int high,
                              unsigned int low, const smt_sort *s);

#if 0
  virtual smt_ast *tuple_create(const expr2tc &structdef);
  virtual smt_ast *tuple_project(const smt_ast *a, const smt_sort *s,
                                 unsigned int field);
  virtual const smt_ast *tuple_update(const smt_ast *a, unsigned int field,
                                      const smt_ast *val);
  virtual const smt_ast *tuple_equality(const smt_ast *a, const smt_ast *val);
  virtual const smt_ast *tuple_ite(const smt_ast *cond, const smt_ast *trueval,
                             const smt_ast *false_val, const smt_sort *sort);

  virtual const smt_ast *tuple_array_create(const type2tc &array_type,
                                            const smt_ast **input_args,
                                            bool const_array,
                                            const smt_sort *domain);
  virtual const smt_ast *tuple_array_select(const smt_ast *a, const smt_sort *s,
                                      const smt_ast *field);
  virtual smt_ast *tuple_array_update(const smt_ast *a, const smt_ast *field,
                                      const smt_ast *val, const smt_sort *s);
  virtual smt_ast *tuple_array_equality(const smt_ast *a, const smt_ast *b);
  virtual smt_ast *tuple_array_ite(const smt_ast *cond, const smt_ast *trueval,
                                   const smt_ast *false_val,
                                   const smt_sort *sort);

  virtual const smt_ast *overflow_arith(const expr2tc &expr);
  virtual smt_ast *overflow_cast(const expr2tc &expr);
  virtual const smt_ast *overflow_neg(const expr2tc &expr);
#endif

  virtual smt_ast *mk_fresh(const smt_sort *s, const std::string &tag);

  // Assert a formula; needs_literal indicates a new literal should be allocated
  // for this assertion (Z3_check_assumptions refuses to deal with assumptions
  // that are not "propositional variables or their negation". So we associate
  // the ast with a literal.
  void assert_formula(const z3::expr &ast);
  virtual void assert_lit(const literalt &l);

  std::string double2string(double d) const;

  std::string get_fixed_point(
	const unsigned width,
    std::string value) const;

  expr2tc bv_get_rec(const Z3_ast bv, const type2tc &type);

  void debug_label_formula(std::string name, const z3::expr &formula);
  void init_addr_space_array(void);

  virtual const std::string solver_text()
  { return "Z3"; }

  virtual tvt l_get(literalt a);

  z3::expr z3_literal(literalt l);

  // Some useful types
public:
  #define z3_smt_downcast(x) static_cast<const z3_smt_ast *>(x)
  class z3_smt_ast : public smt_ast {
  public:
    z3_smt_ast(z3::expr _e, const smt_sort *_s) :
              smt_ast(_s), e(_e) { }
    virtual ~z3_smt_ast() { }
    z3::expr e;
  };

  class z3_smt_sort : public smt_sort {
  public:
    z3_smt_sort(smt_sort_kind i, z3::sort _s, bool is_s = false) : smt_sort(i), s(_s), is_signed(is_s) { }
    virtual ~z3_smt_sort() { }
    z3::sort s;
    bool is_signed;
  };

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
  std::list<unsigned long> total_mem_space;

  // Debug map, for naming pieces of AST and auto-numbering them
  std::map<std::string, unsigned> debug_label_map;

  z3::sort pointer_sort;
  z3::func_decl pointer_decl;

  Z3_context z3_ctx;
};

#endif
