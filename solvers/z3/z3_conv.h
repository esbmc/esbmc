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

#include "z3_prop.h"
#include "z3_capi.h"

#define Z3_UNSAT_CORE_LIMIT 10000

typedef unsigned int uint;

class z3_convt: public prop_convt
{
public:
  z3_convt(bool uw, bool int_encoding, bool smt)
                               :prop_convt(z3_prop),
                                z3_prop(uw)
  {
    if (z3_ctx == NULL) {
      z3_ctx = z3_api.mk_proof_context(uw);
    }

    this->int_encoding = int_encoding;
    this->z3_prop.smtlib = smt;
    this->z3_prop.store_assumptions = (smt || uw);
   s_is_uw = uw;
   total_mem_space = 0;

    Z3_push(z3_ctx);
    z3_prop.z3_ctx = z3_ctx;
    ignoring_expr=true;
    max_core_size=Z3_UNSAT_CORE_LIMIT;

    z3_api.set_z3_ctx(z3_ctx);
    z3_prop.z3_api.set_z3_ctx(z3_ctx);

    init_addr_space_array();
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

  void create_array_type(const typet &type, Z3_type_ast &bv);
  void create_type(const typet &type, Z3_type_ast &bv);
  void create_struct_union_type(const typet &type, bool uni, Z3_type_ast &bv);
  void create_struct_type(const typet &type, Z3_type_ast &bv) {
    create_struct_union_type(type, false, bv);
  }
  void create_union_type(const typet &type, Z3_type_ast &bv) {
    create_struct_union_type(type, true, bv);
  }
  void create_enum_type(Z3_type_ast &bv);
  void create_pointer_type(Z3_type_ast &bv);
  Z3_ast convert_cmp(const exprt &expr);
  Z3_ast convert_eq(const exprt &expr);
  Z3_ast convert_same_object(const exprt &expr);
  Z3_ast convert_is_dynamic_object(const exprt &expr);
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

  void convert_identifier(const std::string &identifier, const typet &type, Z3_ast &bv);
  void convert_bv(const exprt &expr, Z3_ast &bv);

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

  struct eqstr
  {
    bool operator()(const char* s1, const char* s2) const
    {
      return strcmp(s1, s2) == 0;
    }
  };

  typedef hash_map_cont<const exprt, Z3_ast, irep_hash> bv_cachet;
  bv_cachet bv_cache;

  typedef hash_map_cont<const exprt, std::string, irep_hash> z3_cachet;
  z3_cachet z3_cache;

  typedef std::map<std::string, Z3_ast> map_varst;
  map_varst map_vars;

  std::string itos(int i);
  std::string fixed_point(std::string v, unsigned width);
  std::string extract_magnitude(std::string v, unsigned width);
  std::string extract_fraction(std::string v, unsigned width);
  bool is_bv(const typet &type);
  bool check_all_types(const typet &type);
  bool is_ptr(const typet &type);
  bool is_signed(const typet &type);
  bool is_in_cache(const exprt &expr);
  void write_cache(const exprt &expr);
  void read_cache(const exprt &expr, Z3_ast &bv);
  static std::string ascii2int(char ch);
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
  void store_sat_assignments(Z3_model m);
  u_int number_variables_z3, set_to_counter, number_vcs_z3,
	    max_core_size;

  z3_propt z3_prop;
  z3_capi z3_api;

  bool int_encoding, ignoring_expr, equivalence_checking;
  //Z3_ast assumptions[Z3_UNSAT_CORE_LIMIT];
  std::list<Z3_ast> assumptions;
  std::string filename;

  typedef std::map<std::string, unsigned int> union_varst;
  union_varst union_vars;

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
