/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_Z3_CONV_H
#define CPROVER_PROP_Z3_CONV_H

#include <execinfo.h>
#include <stdint.h>

#include <map>
#include <hash_cont.h>
#include <solvers/prop/prop_conv.h>
#include <solvers/flattening/pointer_logic.h>
#include <ext/hash_map>
#include <vector>
#include <string.h>

#include "z3_prop.h"
#include "z3_capi.h"

#define Z3_UNSAT_CORE_LIMIT 10000

typedef unsigned int uint;

class z3_prop_wrappert
{
public:
  z3_prop_wrappert(std::ostream &_out):z3_prop(_out) { }

protected:
  z3_propt z3_prop;
};

class z3_convt:protected z3_prop_wrappert, public prop_convt
{
public:
  z3_convt(std::ostream &_out, bool relevancy, bool uw):z3_prop_wrappert(_out),
                                prop_convt(z3_prop)
  {
    if (z3_ctx == NULL) {
      if (relevancy) {
        z3_ctx = z3_api.mk_proof_context(false, uw);
      } else {
        z3_ctx = z3_api.mk_proof_context(true, uw);
      }
    }

   s_is_uw = uw;
   s_relevancy = relevancy;

    Z3_push(z3_ctx);
    z3_prop.z3_ctx = z3_ctx;
    this->uw = uw;
    ignoring_expr=true;
    max_core_size=Z3_UNSAT_CORE_LIMIT;
  }

  virtual ~z3_convt();
  Z3_lbool check2_z3_properties(void);
  void set_z3_encoding(bool enc);
  void set_smtlib(bool smt);
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

  bool create_array_type(const typet &type, Z3_type_ast &bv);
  bool create_type(const typet &type, Z3_type_ast &bv);
  bool create_struct_union_type(const typet &type, bool uni, Z3_type_ast &bv);
  bool create_struct_type(const typet &type, Z3_type_ast &bv) {
    return create_struct_union_type(type, false, bv);
  }
  bool create_union_type(const typet &type, Z3_type_ast &bv) {
    return create_struct_union_type(type, true, bv);
  }
  bool create_enum_type(Z3_type_ast &bv);
  bool create_pointer_type(const typet &type, Z3_type_ast &bv);
  Z3_ast convert_lt(const exprt &expr);
  Z3_ast convert_gt(const exprt &expr);
  Z3_ast convert_le(const exprt &expr);
  Z3_ast convert_ge(const exprt &expr);
  Z3_ast convert_eq(const exprt &expr);
  Z3_ast convert_invalid(const exprt &expr);
  Z3_ast convert_same_object(const exprt &expr);
  Z3_ast convert_dynamic_object(const exprt &expr);
  Z3_ast convert_overflow_sum_sub_mul(const exprt &expr);
  Z3_ast convert_overflow_unary(const exprt &expr);
  Z3_ast convert_overflow_typecast(const exprt &expr);
  Z3_ast convert_rest_index(const exprt &expr);
  Z3_ast convert_rest_member(const exprt &expr);
  Z3_ast convert_memory_leak(const exprt &expr);
  bool convert_rel(const exprt &expr, Z3_ast &bv);
  bool convert_typecast(const exprt &expr, Z3_ast &bv);
  bool convert_typecast_bool(const exprt &expr, Z3_ast &bv);
  bool convert_typecast_fixedbv_nonint(const exprt &expr, Z3_ast &bv);
  bool convert_typecast_ints_ptrs(const exprt &expr, Z3_ast &bv);
  bool convert_typecast_struct(const exprt &expr, Z3_ast &bv);
  bool convert_typecast_enum(const exprt &expr, Z3_ast &bv);
  bool convert_struct_union(const exprt &expr, Z3_ast &bv);
  bool convert_z3_pointer(const exprt &expr, std::string symbol, Z3_ast &bv);
  bool convert_zero_string(const exprt &expr, Z3_ast &bv);
  bool convert_array(const exprt &expr, Z3_ast &bv);
  bool convert_constant(const exprt &expr, Z3_ast &bv);
  bool convert_bitwise(const exprt &expr, Z3_ast &bv);
  bool convert_unary_minus(const exprt &expr, Z3_ast &bv);
  bool convert_if(const exprt &expr, Z3_ast &bv);
  bool convert_logical_ops(const exprt &expr, Z3_ast &bv);
  bool convert_logical_not(const exprt &expr, Z3_ast &bv);
  bool convert_equality(const exprt &expr, Z3_ast &bv);
  bool convert_add_sub(const exprt &expr, Z3_ast &bv);
  bool convert_div(const exprt &expr, Z3_ast &bv);
  bool convert_mod(const exprt &expr, Z3_ast &bv);
  bool convert_mul(const exprt &expr, Z3_ast &bv);
  bool convert_pointer(const exprt &expr, Z3_ast &bv);
  bool convert_array_of(const exprt &expr, Z3_ast &bv);
  bool convert_index(const exprt &expr, Z3_ast &bv);
  bool convert_shift(const exprt &expr, Z3_ast &bv);
  bool convert_abs(const exprt &expr, Z3_ast &bv);
  bool convert_with(const exprt &expr, Z3_ast &bv);
  bool convert_bitnot(const exprt &expr, Z3_ast &bv);
  bool convert_same_object(const exprt &expr, Z3_ast &bv);
  bool select_pointer_offset(const exprt &expr, Z3_ast &bv);
  bool convert_member(const exprt &expr, Z3_ast &bv);
  bool convert_pointer_object(const exprt &expr, Z3_ast &bv);
  bool convert_zero_string_length(const exprt &expr, Z3_ast &bv);
  bool select_pointer_value(Z3_ast object, Z3_ast offset, Z3_ast &bv);
  bool convert_is_dynamic_object(const exprt &expr, Z3_ast &bv);
  bool convert_byte_update(const exprt &expr, Z3_ast &bv);
  bool convert_byte_extract(const exprt &expr, Z3_ast &bv);
  bool convert_isnan(const exprt &expr, Z3_ast &bv);
  bool convert_z3_expr(const exprt &expr, Z3_ast &bv);

  bool convert_identifier(const std::string &identifier, const typet &type, Z3_ast &bv);
  bool convert_bv(const exprt &expr, Z3_ast &bv);

  std::string double2string(double d) const;

  std::string get_fixed_point(
	const unsigned width,
    std::string value) const;

  exprt bv_get_rec(
	const Z3_ast &bv,
    std::vector<exprt> &unknown,
    const bool cache,
    const typet &type) const;

  void fill_vector(
    const Z3_ast &bv,
    std::vector<exprt> &unknown,
    const typet &type) const;

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
  bool write_cache(const exprt &expr);
  const typet select_pointer(const typet &type);
  bool read_cache(const exprt &expr, Z3_ast &bv);
  static std::string ascii2int(char ch);
  void print_data_types(Z3_ast operand0, Z3_ast operand1);
  void print_location(const exprt &expr);
  void show_bv_size(Z3_ast operand);
  Z3_ast convert_number(int64_t value, u_int width, bool type);
  Z3_ast convert_number_int(int64_t value, u_int width, bool type);
  Z3_ast convert_number_bv(int64_t value, u_int width, bool type);
  void generate_assumptions(const exprt &expr, const Z3_ast &result);
  void store_sat_assignments(Z3_model m);
  u_int number_variables_z3, set_to_counter, number_vcs_z3,
	    max_core_size;
  z3_capi z3_api;
  bool int_encoding, ignoring_expr, equivalence_checking, uw;
  //Z3_ast assumptions[Z3_UNSAT_CORE_LIMIT];
  std::vector<Z3_ast> assumptions;
  std::string filename;

  typedef std::map<std::string, unsigned int> union_varst;
  union_varst union_vars;

  class conv_error {
    void *backtrace_ptrs[50];
    char **backtrace_syms;
    int num_frames;
    std::string msg;
    irept irep;

    conv_error(std::string msg, irept irep) {
      this->msg = msg;
      this->irep = irep;
      num_frames = backtrace(backtrace_ptrs, 50);
      backtrace_syms = backtrace_symbols(backtrace_ptrs, num_frames);
      return;
    }

    std::string to_string(void) {
      std::string msg;
      msg = "Encountered Z3 conversion error: \"" + msg + "\" at:\n";
      for (int i = 0; i < num_frames; i++) {
        msg += backtrace_syms[i];
        msg += "\n";
      }

      if (num_frames == 0)
        msg += "(couldn't get a backtrace)\n";

      return msg;
    }
  };

public:
  static Z3_context z3_ctx;
  static bool s_is_uw;
  static bool s_relevancy;
  static unsigned int num_ctx_ileaves; // Number of ileaves z3_ctx has handled
};

#endif
