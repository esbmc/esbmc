/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_Z3_CONV_H
#define CPROVER_PROP_Z3_CONV_H

#ifdef __linux__
#include <execinfo.h>
#endif
#include <stdint.h>

#include <map>
#include <set>
#include <hash_cont.h>
#include <solvers/prop/prop_conv.h>
#include <solvers/flattening/pointer_logic.h>
#include <vector>
#include <string.h>
#include <decision_procedure.h>
#include <irep2.h>

#include "z3_prop.h"
#include "z3_capi.h"

#define Z3_UNSAT_CORE_LIMIT 10000

typedef unsigned int uint;

class z3_convt: public prop_convt
{
public:
  z3_convt(bool uw, bool int_encoding, bool smt, bool is_cpp)
                               :prop_convt(z3_prop),
                                z3_prop(uw, *this)
  {
    if (z3_ctx == NULL) {
      z3_ctx = z3_api.mk_proof_context(uw);
    }

    this->int_encoding = int_encoding;
    this->z3_prop.smtlib = smt;
    this->z3_prop.store_assumptions = (smt || uw);
    s_is_uw = uw;
    total_mem_space = 0;
    model = NULL;
    array_of_count = 0;

    Z3_push(z3_ctx);
    z3_prop.z3_ctx = z3_ctx;
    ignoring_expr=true;
    max_core_size=Z3_UNSAT_CORE_LIMIT;

    z3_api.set_z3_ctx(z3_ctx);
    z3_prop.z3_api.set_z3_ctx(z3_ctx);

    init_addr_space_array();

    // Pick a modelling array to shoehorn initialization data into. Because
    // we don't yet have complete data for whether pointers are dynamic or not,
    // this is the one modelling array that absolutely _has_ to be initialized
    // to false for each element, which is going to be shoved into
    // convert_identifier_pointer.
    if (is_cpp) {
      dyn_info_arr_name = "cpp::__ESBMC_is_dynamic&0#1";
    } else {
      dyn_info_arr_name = "c::__ESBMC_is_dynamic&0#1";
    }
  }

  virtual ~z3_convt();
  virtual decision_proceduret::resultt dec_solve(void);
  Z3_lbool check2_z3_properties(void);
  bool get_z3_encoding(void) const;
  void set_filename(std::string file);
  void set_z3_ecp(bool ecp);
  uint get_z3_core_size(void);
  uint get_z3_number_of_assumptions(void);
  void set_z3_core_size(uint val);

  // overloading
  virtual exprt get(const exprt &expr) const;

  u_int get_number_variables_z3(void);
  u_int get_number_vcs_z3(void);

private:
  virtual literalt convert_rest(const exprt &expr);
  virtual void set_to(const exprt &expr, bool value);
  bool assign_z3_expr(const exprt expr);
  u_int convert_member_name(const exprt &lhs, const exprt &rhs);

  void create_array_type(const typet &type, Z3_type_ast &bv) const;
  void create_type(const typet &type, Z3_type_ast &bv) const;
  void create_struct_union_type(const typet &type, bool uni, Z3_type_ast &bv) const;
  void create_struct_type(const typet &type, Z3_type_ast &bv) const {
    create_struct_union_type(type, false, bv);
  }
  void create_union_type(const typet &type, Z3_type_ast &bv) const {
    create_struct_union_type(type, true, bv);
  }
  void create_enum_type(Z3_type_ast &bv) const;
  void create_pointer_type(Z3_type_ast &bv) const;
  Z3_ast convert_cmp(const exprt &expr);
  Z3_ast convert_eq(const exprt &expr);
  Z3_ast convert_same_object(const exprt &expr);
  Z3_ast convert_invalid_object(const exprt &expr);
  Z3_ast convert_overflow_sum_sub_mul(const exprt &expr);
  Z3_ast convert_overflow_unary(const exprt &expr);
  Z3_ast convert_overflow_typecast(const exprt &expr);
  Z3_ast convert_memory_leak(const exprt &expr);
  Z3_ast convert_width(const exprt &expr);
  void convert_typecast(const exprt &expr, Z3_ast &bv);
  void convert_typecast_bool(const exprt &expr, Z3_ast &bv);
  void convert_typecast_fixedbv_nonint(const exprt &expr, Z3_ast &bv);
  void convert_typecast_to_ints(const exprt &expr, Z3_ast &bv);
  void convert_typecast_to_ptr(const exprt &expr, Z3_ast &bv);
  void convert_typecast_from_ptr(const exprt &expr, Z3_ast &bv);
  void convert_typecast_struct(const exprt &expr, Z3_ast &bv);
  void convert_typecast_enum(const exprt &expr, Z3_ast &bv);
  void convert_struct_union(const exprt &expr, Z3_ast &bv);
  void convert_identifier_pointer(const exprt &expr, std::string symbol,
                                  Z3_ast &bv);
  void convert_zero_string(const exprt &expr, Z3_ast &bv);
  void convert_array(const exprt &expr, Z3_ast &bv);
  void convert_constant(const exprt &expr, Z3_ast &bv);
  void convert_bitwise(const exprt &expr, Z3_ast &bv);
  void convert_unary_minus(const exprt &expr, Z3_ast &bv);
  void convert_if(const exprt &expr, Z3_ast &bv);
  void convert_logical_ops(const exprt &expr, Z3_ast &bv);
  void convert_logical_not(const exprt &expr, Z3_ast &bv);
  void convert_equality(const exprt &expr, Z3_ast &bv);
  void convert_pointer_arith(const exprt &expr, Z3_ast &bv);
  void convert_add_sub(const exprt &expr, Z3_ast &bv);
  void convert_div(const exprt &expr, Z3_ast &bv);
  void convert_mod(const exprt &expr, Z3_ast &bv);
  void convert_mul(const exprt &expr, Z3_ast &bv);
  void convert_address_of(const exprt &expr, Z3_ast &bv);
  void convert_array_of(const exprt &expr, Z3_ast &bv);
  void convert_index(const exprt &expr, Z3_ast &bv);
  void convert_shift(const exprt &expr, Z3_ast &bv);
  void convert_abs(const exprt &expr, Z3_ast &bv);
  void convert_with(const exprt &expr, Z3_ast &bv);
  void convert_bitnot(const exprt &expr, Z3_ast &bv);
  void select_pointer_offset(const exprt &expr, Z3_ast &bv);
  void convert_member(const exprt &expr, Z3_ast &bv);
  void convert_pointer_object(const exprt &expr, Z3_ast &bv);
  void convert_zero_string_length(const exprt &expr, Z3_ast &bv);
  void select_pointer_value(Z3_ast object, Z3_ast offset, Z3_ast &bv);
  void convert_byte_update(const exprt &expr, Z3_ast &bv);
  void convert_byte_extract(const exprt &expr, Z3_ast &bv);
  void convert_isnan(const exprt &expr, Z3_ast &bv);
  void convert_z3_expr(const exprt &expr, Z3_ast &bv);

  void convert_bv(const exprt &expr, Z3_ast &bv);

  void convert_identifier(const std::string &identifier, const typet &type, Z3_ast &bv);

  typedef Z3_ast (*ast_convert_calltype)(Z3_context ctx, Z3_ast op1, Z3_ast op2);
  typedef Z3_ast (*ast_convert_multiargs)(Z3_context ctx, unsigned int numargs,
                                          Z3_ast const args[]);

  void convert_rel(const rel2t &rel, ast_convert_calltype intmode,
                   ast_convert_calltype signedbv,
                   ast_convert_calltype unsignedbv,
                   void *&_bv);
  void convert_logic_2ops(const logical_2ops2t &log,
                          ast_convert_calltype converter,
                          ast_convert_multiargs bulkconverter,
                          void *&_bv);
  void convert_binop(const binops2t &log,
                    ast_convert_calltype converter,
                    void *&_bv);
  void convert_arith2ops(const arith_2op2t &log,
                         ast_convert_calltype bvconvert,
                         ast_convert_multiargs intmodeconvert,
                         void *&_bv);
  void convert_shift(const expr2t &shift, const expr2t &part1,
                     const expr2t &part2, ast_convert_calltype convert,
                     void *&_bv);

  virtual void convert_smt_expr(const symbol2t &sym, void *&bv);
  virtual void convert_smt_expr(const constant_int2t &sym, void *&bv);
  virtual void convert_smt_expr(const constant_datatype2t &strt, void *&bv);
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
  virtual void convert_smt_expr(const bitand2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitor2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitxor2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitnand2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitnor2t &bitval, void *&bv);
  virtual void convert_smt_expr(const bitnxor2t &bitval, void *&bv);
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

  virtual void convert_smt_type(const bool_type2t &type, void *&bv) const;
  virtual void convert_smt_type(const bv_type2t &type, void *&bv) const;
  virtual void convert_smt_type(const array_type2t &type, void *&bv) const;
  virtual void convert_smt_type(const pointer_type2t &type, void *&bv) const;
  virtual void convert_smt_type(const struct_union_type2t &type, void *&bv) const;
  virtual void convert_smt_type(const fixedbv_type2t &type, void *&bv) const;

  // Assert a formula; needs_literal indicates a new literal should be allocated
  // for this assertion (Z3_check_assumptions refuses to deal with assumptions
  // that are not "propositional variables or their negation". So we associate
  // the ast with a literal.
  void assert_formula(Z3_ast ast, bool needs_literal = true);
  void assert_literal(literalt l, Z3_ast ast);

  void get_type_width(const typet &t, unsigned &width) const;

  std::string double2string(double d) const;

  std::string get_fixed_point(
	const unsigned width,
    std::string value) const;

  exprt bv_get_rec(const Z3_ast bv, const typet &type) const;

  pointer_logict pointer_logic;

  typedef hash_map_cont<const exprt, Z3_ast, irep_hash> bv_cachet;
  bv_cachet bv_cache;

  std::string itos(int i);
  std::string fixed_point(std::string v, unsigned width);
  std::string extract_magnitude(std::string v, unsigned width);
  std::string extract_fraction(std::string v, unsigned width);
  bool is_bv(const typet &type);
  bool is_ptr(const typet &type);
  bool is_signed(const typet &type);
  void print_data_types(Z3_ast operand0, Z3_ast operand1);
  void print_location(const exprt &expr);
  void debug_label_formula(std::string name, Z3_ast formula);
  void show_bv_size(Z3_ast operand);
  Z3_ast convert_number(int64_t value, u_int width, bool type);
  Z3_ast convert_number_int(int64_t value, u_int width, bool type);
  Z3_ast convert_number_bv(int64_t value, u_int width, bool type);
  void bump_addrspace_array(unsigned int idx, Z3_ast val);
  std::string get_cur_addrspace_ident(void);
  void generate_assumptions(const exprt &expr, const Z3_ast &result);
  void link_syms_to_literals(void);
  void finalize_pointer_chain(void);
  void init_addr_space_array(void);
  u_int number_variables_z3, set_to_counter, number_vcs_z3,
	    max_core_size;

  Z3_model model; // Model of satisfying program.

  z3_propt z3_prop;
  z3_capi z3_api;

  bool int_encoding, ignoring_expr, equivalence_checking;
  //Z3_ast assumptions[Z3_UNSAT_CORE_LIMIT];
  std::list<Z3_ast> assumptions;
  std::string filename;

  typedef std::map<std::string, unsigned int> union_varst;
  union_varst union_vars;

  unsigned int array_of_count;
  irep_idt dyn_info_arr_name;

  // Array of obj ID -> address range tuples
  unsigned int addr_space_sym_num;
  Z3_sort addr_space_tuple_sort;
  Z3_sort addr_space_arr_sort;
  std::map<unsigned,unsigned> addr_space_data; // Obj id, size
  unsigned long total_mem_space;

  // Debug map, for naming pieces of AST and auto-numbering them
  std::map<std::string, unsigned> debug_label_map;

public:
  class conv_error {
    void *backtrace_ptrs[50];
    char **backtrace_syms;
    int num_frames;
    std::string msg;
    irept irep;

  public:
    conv_error(std::string msg, irept irep) {
      this->msg = msg;
      this->irep = irep;
#ifndef _WIN32
      num_frames = backtrace(backtrace_ptrs, 50);
      backtrace_syms = backtrace_symbols(backtrace_ptrs, num_frames);
#else
      num_frames = 0;
      backtrace_syms = NULL;
#endif
      return;
    }

    std::string to_string(void) {
      std::string out;
      out = "Encountered Z3 conversion error: \"" + msg + "\" at:\n";
      for (int i = 0; i < num_frames; i++) {
        out += backtrace_syms[i];
        out += "\n";
      }

      if (num_frames == 0)
        out += "(couldn't get a backtrace)\n";

      out += "For irep:" + irep.pretty(0);

      return out;
    }
  };

public:
  static Z3_context z3_ctx;
  static bool s_is_uw;
  static unsigned int num_ctx_ileaves; // Number of ileaves z3_ctx has handled
};

#endif
