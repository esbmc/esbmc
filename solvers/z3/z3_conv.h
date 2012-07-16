/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_Z3_CONV_H
#define CPROVER_PROP_Z3_CONV_H

#include <irep2.h>

#include <stdint.h>

#include <map>
#include <set>
#include <hash_cont.h>
#include <solvers/prop/prop_conv.h>
#include <solvers/prop/pointer_logic.h>
#include <vector>
#include <string.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include "z3++.h"
#include "z3_capi.h"

#define Z3_UNSAT_CORE_LIMIT 10000

typedef unsigned int uint;

class z3_convt: public prop_convt
{
public:
  z3_convt(bool uw, bool int_encoding, bool smt, bool is_cpp);
  virtual ~z3_convt();
private:
  void intr_push_ctx(void);
  void intr_pop_ctx(void);
public:
  virtual void push_ctx(void);
  virtual void pop_ctx(void);
  virtual void soft_push_ctx(void);
  virtual void soft_pop_ctx(void);
  virtual prop_convt::resultt dec_solve(void);
  Z3_lbool check2_z3_properties(void);
  bool get_z3_encoding(void) const;
  void set_filename(std::string file);
  uint get_z3_core_size(void);
  uint get_z3_number_of_assumptions(void);
  void set_z3_core_size(uint val);

  // overloading
  virtual expr2tc get(const expr2tc &expr) const;

private:
  virtual literalt convert_expr(const expr2tc &expr);
  virtual void set_to(const expr2tc &expr, bool value);
  bool assign_z3_expr(const exprt expr);
  u_int convert_member_name(const exprt &lhs, const exprt &rhs);

  void create_pointer_type(Z3_sort &bv) const;
  void convert_type(const type2tc &type, Z3_sort &outtype);

  void convert_bv(const expr2tc &expr, Z3_ast &bv);

  void convert_typecast_bool(const typecast2t &cast, Z3_ast &bv);
  void convert_typecast_fixedbv_nonint(const typecast2t &cast, Z3_ast &bv);
  void convert_typecast_to_ints(const typecast2t &cast, Z3_ast &bv);
  void convert_typecast_to_ptr(const typecast2t &castj, Z3_ast &bv);
  void convert_typecast_from_ptr(const typecast2t &cast, Z3_ast &bv);
  void convert_typecast_struct(const typecast2t &cast, Z3_ast &bv);

  void convert_identifier_pointer(const expr2tc &expr, std::string symbol,
                                  Z3_ast &bv);

  typedef Z3_ast (*ast_convert_calltype)(Z3_context ctx, Z3_ast op1, Z3_ast op2);
  typedef Z3_ast (*ast_convert_multiargs)(Z3_context ctx, unsigned int numargs,
                                          Z3_ast const args[]);

  void convert_rel(const expr2tc &side1, const expr2tc &side2,
                   ast_convert_calltype intmode,
                   ast_convert_calltype signedbv,
                   ast_convert_calltype unsignedbv,
                   void *&_bv);
  void convert_logic_2ops(const expr2tc &side1, const expr2tc &side2,
                          ast_convert_calltype converter,
                          ast_convert_multiargs bulkconverter,
                          void *&_bv);
  void convert_binop(const expr2tc &side1, const expr2tc &side2,
                    const type2tc &type, ast_convert_calltype converter,
                    void *&_bv);
  void convert_arith2ops(const expr2tc &side1, const expr2tc &side2,
                         ast_convert_calltype bvconvert,
                         ast_convert_multiargs intmodeconvert,
                         void *&_bv);
  void convert_shift(const expr2t &shift, const expr2tc &part1,
                     const expr2tc &part2, ast_convert_calltype convert,
                     void *&_bv);

  void convert_pointer_arith(expr2t::expr_ids id, const expr2tc &side1,
                             const expr2tc &side2,
                             const type2tc &type, Z3_ast &bv);

  void convert_struct_union(const std::vector<expr2tc> &members,
                            const std::vector<type2tc> &member_types,
                            const type2tc &type, bool is_union, void *&_bv);

  virtual void convert_smt_expr(const symbol2t &sym, void *&bv);
  virtual void convert_smt_expr(const constant_int2t &sym, void *&bv);
  virtual void convert_smt_expr(const constant_fixedbv2t &sym, void *&bv);
  virtual void convert_smt_expr(const constant_bool2t &b, void *&bv);
  virtual void convert_smt_expr(const constant_struct2t &strt, void *&bv);
  virtual void convert_smt_expr(const constant_union2t &strt, void *&bv);
  virtual void convert_smt_expr(const constant_array2t &array, void *&bv);
  virtual void convert_smt_expr(const constant_array_of2t &array, void *&bv);
  virtual void convert_smt_expr(const constant_string2t &str, void *&bv);
  virtual void convert_smt_expr(const if2t &ifirep, void *&bv);
  virtual void convert_smt_expr(const equality2t &equality, void *&bv);
  virtual void convert_smt_expr(const notequal2t &notequal, void *&bv);
  virtual void convert_smt_expr(const lessthan2t &lessthan, void *&bv);
  virtual void convert_smt_expr(const greaterthan2t &greaterthan, void *&bv);
  virtual void convert_smt_expr(const lessthanequal2t &le, void *&bv);
  virtual void convert_smt_expr(const greaterthanequal2t &le, void *&bv);
  virtual void convert_smt_expr(const not2t &notval, void *&bv);
  virtual void convert_smt_expr(const and2t &andval, void *&bv);
  virtual void convert_smt_expr(const or2t &orval, void *&bv);
  virtual void convert_smt_expr(const xor2t &xorval, void *&bv);
  virtual void convert_smt_expr(const implies2t &implies, void *&bv);
  virtual void convert_smt_expr(const bitand2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitor2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitxor2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitnand2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitnor2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitnxor2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitnot2t &bitval, void *&bv);
  virtual void convert_smt_expr(const lshr2t &bitval, void *&bv);
  virtual void convert_smt_expr(const neg2t &neg, void *&bv);
  virtual void convert_smt_expr(const abs2t &abs, void *&bv);
  virtual void convert_smt_expr(const add2t &add, void *&bv);
  virtual void convert_smt_expr(const sub2t &sub, void *&bv);
  virtual void convert_smt_expr(const mul2t &mul, void *&bv);
  virtual void convert_smt_expr(const div2t &mul, void *&bv);
  virtual void convert_smt_expr(const modulus2t &mod, void *&bv);
  virtual void convert_smt_expr(const shl2t &shl, void *&bv);
  virtual void convert_smt_expr(const ashr2t &ashr, void *&bv);
  virtual void convert_smt_expr(const same_object2t &same, void *&bv);
  virtual void convert_smt_expr(const pointer_offset2t &offs, void *&bv);
  virtual void convert_smt_expr(const pointer_object2t &obj, void *&bv);
  virtual void convert_smt_expr(const address_of2t &obj, void *&bv);
  virtual void convert_smt_expr(const byte_extract2t &data, void *&bv);
  virtual void convert_smt_expr(const byte_update2t &data, void *&bv);
  virtual void convert_smt_expr(const with2t &with, void *&bv);
  virtual void convert_smt_expr(const member2t &member, void *&bv);
  virtual void convert_smt_expr(const typecast2t &cast, void *&bv);
  virtual void convert_smt_expr(const index2t &index, void *&bv);
  virtual void convert_smt_expr(const zero_string2t &zstr, void *&bv);
  virtual void convert_smt_expr(const zero_length_string2t &s, void *&bv);
  virtual void convert_smt_expr(const isnan2t &isnan, void *&bv);
  virtual void convert_smt_expr(const overflow2t &overflow, void *&bv);
  virtual void convert_smt_expr(const overflow_cast2t &ocast, void *&arg);
  virtual void convert_smt_expr(const overflow_neg2t &neg, void *&arg);

  virtual void convert_smt_type(const bool_type2t &type, void *&bv);
  virtual void convert_smt_type(const unsignedbv_type2t &type, void *&bv);
  virtual void convert_smt_type(const signedbv_type2t &type, void *&bv);
  virtual void convert_smt_type(const array_type2t &type, void *&bv);
  virtual void convert_smt_type(const pointer_type2t &type, void *&bv);
  virtual void convert_smt_type(const struct_type2t &type, void *&bv);
  virtual void convert_smt_type(const union_type2t &type, void *&bv);
  virtual void convert_smt_type(const fixedbv_type2t &type, void *&bv);

  void convert_struct_union_type(const std::vector<type2tc> &members,
                                 const std::vector<irep_idt> &member_names,
                                 const irep_idt &name, bool uni, void *&_bv);

  // Assert a formula; needs_literal indicates a new literal should be allocated
  // for this assertion (Z3_check_assumptions refuses to deal with assumptions
  // that are not "propositional variables or their negation". So we associate
  // the ast with a literal.
  void assert_formula(Z3_ast ast);

  std::string double2string(double d) const;

  std::string get_fixed_point(
	const unsigned width,
    std::string value) const;

  expr2tc bv_get_rec(const Z3_ast bv, const type2tc &type) const;

  std::list<pointer_logict> pointer_logic;

  // Types for bv_cache.

  struct bv_cache_entryt {
    const expr2tc val;
    Z3_ast output;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    bv_cache_entryt,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(bv_cache_entryt, const expr2tc, val)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(bv_cache_entryt, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > bv_cachet;

  // Types for union map.
  struct union_var_mapt {
    std::string ident;
    unsigned int idx;
    unsigned int level;
  };

  typedef boost::multi_index_container<
    union_var_mapt,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(union_var_mapt, std::string, ident)
      >,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(union_var_mapt, unsigned int, level),
        std::greater<unsigned int>
      >
    >
  > union_varst;

  bv_cachet bv_cache;
  union_varst union_vars;
  typedef hash_map_cont<const type2tc, Z3_sort, type2_hash> sort_cachet;
  sort_cachet sort_cache;

  std::string itos(int i);
  std::string fixed_point(std::string v, unsigned width);
  std::string extract_magnitude(std::string v, unsigned width);
  std::string extract_fraction(std::string v, unsigned width);
  void debug_label_formula(std::string name, Z3_ast formula);
  Z3_ast convert_number(int64_t value, u_int width, bool type);
  Z3_ast convert_number_int(int64_t value, u_int width, bool type);
  Z3_ast convert_number_bv(int64_t value, u_int width, bool type);
  void bump_addrspace_array(unsigned int idx, Z3_ast val);
  std::string get_cur_addrspace_ident(void);
  void finalize_pointer_chain(unsigned int objnum);
  void init_addr_space_array(void);

  virtual literalt land(literalt a, literalt b);
  virtual literalt lor(literalt a, literalt b);
  virtual literalt land(const bvt &bv);
  virtual literalt lor(const bvt &bv);
  virtual literalt lnot(literalt a);
  virtual literalt limplies(literalt a, literalt b);
  virtual literalt new_variable();
  virtual uint64_t get_no_variables() const { return no_variables; }
  virtual void set_no_variables(uint64_t no) { no_variables = no; }
  virtual void lcnf(const bvt &bv);

  virtual const std::string solver_text()
  { return "Z3"; }

  virtual tvt l_get(literalt a) const;

  Z3_ast z3_literal(literalt l);

  bool process_clause(const bvt &bv, bvt &dest);

  u_int number_variables_z3, set_to_counter, number_vcs_z3,
	    max_core_size;

  Z3_model model; // Model of satisfying program.

  z3_capi z3_api;

  unsigned int level_ctx;
  bool int_encoding, smtlib, store_assumptions, uw;
  std::string filename;

  unsigned int array_of_count;
  std::string dyn_info_arr_name;

  uint64_t no_variables;
  std::list<Z3_ast> assumpt;
  std::list<std::list<Z3_ast>::iterator> assumpt_ctx_stack;

  // Array of obj ID -> address range tuples
  std::list<unsigned int> addr_space_sym_num;
  Z3_sort addr_space_tuple_sort;
  Z3_sort addr_space_arr_sort;
  std::list<std::map<unsigned, unsigned>> addr_space_data; // Obj id, size
  std::list<unsigned long> total_mem_space;

  // Debug map, for naming pieces of AST and auto-numbering them
  std::map<std::string, unsigned> debug_label_map;

public:
  class conv_error {
    std::string msg;

  public:
    conv_error(std::string msg) {
      this->msg = msg;
      return;
    }

    std::string to_string(void) {
      std::string out;
      out = "Encountered Z3 conversion error: \"" + msg + "\"\n";
      return out;
    }
  };

public:
  Z3_context z3_ctx;
  static bool s_is_uw;
};

#endif
