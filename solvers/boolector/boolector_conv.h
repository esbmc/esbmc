/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_BOOLECTOR_CONV_H
#define CPROVER_PROP_BOOLECTOR_CONV_H

#include <hash_cont.h>
#include <solvers/prop/prop_conv.h>
#include <solvers/flattening/pointer_logic.h>

#include "boolector_prop.h"

class boolector_prop_wrappert
{
public:
  boolector_prop_wrappert(std::ostream &_out):boolector_prop(_out) { }

protected:
  boolector_propt boolector_prop;
};

class boolector_convt:protected boolector_prop_wrappert, public prop_convt
{
public:
  boolector_convt(std::ostream &_out);
  ~boolector_convt();

  void set_filename(std::string file);
  unsigned int get_number_variables_boolector(void);
  int check_boolector_properties(void);
  // overloading
  virtual exprt get(const exprt &expr) const;

protected:
  literalt convert_rest(const exprt &expr);
  bool convert_boolector_expr(const exprt &expr, BtorExp* &bv);
  void set_to(const exprt &expr, bool value);
  bool assign_boolector_expr(const exprt expr);
  bool convert_rel(const exprt &expr, BtorExp* &bv);
  bool convert_typecast(const exprt &expr, BtorExp* &bv);
  bool convert_struct(const exprt &expr, BtorExp* &bv);
  bool convert_union(const exprt &expr, BtorExp* &bv);
  bool convert_constant(const exprt &expr, BtorExp* &bv);
  bool convert_concatenation(const exprt &expr, BtorExp* &bv);
  bool convert_bitwise(const exprt &expr, BtorExp* &bv);
  bool convert_bitnot(const exprt &expr, BtorExp* &bv);
  bool convert_unary_minus(const exprt &expr, BtorExp* &bv);
  bool convert_if(const exprt &expr, BtorExp* &bv);
  bool convert_logical_ops(const exprt &expr, BtorExp* &bv);
  bool convert_logical_not(const exprt &expr, BtorExp* &bv);
  bool convert_equality(const exprt &expr, BtorExp* &bv);
  bool convert_add(const exprt &expr, BtorExp* &bv);
  bool convert_sub(const exprt &expr, BtorExp* &bv);
  bool convert_div(const exprt &expr, BtorExp* &bv);
  bool convert_mod(const exprt &expr, BtorExp* &bv);
  bool convert_mul(const exprt &expr, BtorExp* &bv);
  bool convert_pointer(const exprt &expr, BtorExp* &bv);
  bool convert_array_of(const exprt &expr, BtorExp* &bv);
  bool convert_array_of_array(const std::string identifier, const exprt &expr, BtorExp* &bv);
  bool convert_index(const exprt &expr, BtorExp* &bv);
  bool convert_shift(const exprt &expr, BtorExp* &bv);
  bool convert_with(const exprt &expr, BtorExp* &bv);
  bool convert_extractbit(const exprt &expr, BtorExp* &bv);
  bool convert_member(const exprt &expr, BtorExp* &bv);
  unsigned int convert_member_name(const exprt &lhs, const exprt &rhs);
  bool convert_object(const exprt &expr, BtorExp* &bv);
  bool select_pointer_offset(const exprt &expr, BtorExp* &bv);
  bool convert_invalid_pointer(const exprt &expr, BtorExp* &bv);
  bool convert_pointer_object(const exprt &expr, BtorExp* &bv);
  bool convert_zero_string(const exprt &expr, BtorExp* &bv);
  BtorExp* convert_overflow_sum(const exprt &expr);
  BtorExp* convert_overflow_sub(const exprt &expr);
  BtorExp* convert_overflow_mul(const exprt &expr);
  BtorExp* convert_overflow_typecast(const exprt &expr);
  BtorExp* convert_overflow_unary(const exprt &expr);
  bool convert_boolector_pointer(const exprt &expr, BtorExp* &bv);
  bool convert_array(const exprt &expr, BtorExp* &bv);
  bool select_pointer_value(BtorExp* object, BtorExp* offset, BtorExp* &bv);
  bool convert_abs(const exprt &expr, BtorExp* &bv);
  bool convert_pointer_offset(unsigned bits, BtorExp* &bv);

  pointer_logict pointer_logic;

  typedef hash_map_cont<const exprt, std::string, irep_hash> pointer_cachet;
  pointer_cachet pointer_cache;

  typedef hash_map_cont<const exprt, BtorExp*, irep_hash> bv_cachet;
  bv_cachet bv_cache;

  exprt bv_get_rec(
  BtorExp *bv,
  std::vector<exprt> &unknown,
  const bool cache,
  const typet &type) const;

private:
  bool is_ptr(const typet &type);
  bool check_all_types(const typet &type);
  bool is_signed(const typet &type);
  bool is_set(void);
  void write_cache(const exprt &expr);
  bool convert_constant_array(const exprt &expr, BtorExp* &bv);
  bool convert_bv(const exprt &expr, BtorExp* &bv);
  bool convert_shift_constant(const exprt &expr, unsigned int wop0, unsigned int wop1, BtorExp* &bv);
  bool create_boolector_array(const typet &type, std::string identifier, BtorExp* &bv);
  bool read_cache(const exprt &expr, BtorExp* &bv);
  BtorExp* convert_eq(const exprt &expr);
  BtorExp* convert_ge(const exprt &expr);
  BtorExp* convert_le(const exprt &expr);
  BtorExp* convert_gt(const exprt &expr);
  BtorExp* convert_lt(const exprt &expr);
  BtorExp* convert_dynamic_object(const exprt &expr);
  BtorExp* convert_same_object(const exprt &expr);
  BtorExp* convert_invalid(const exprt &expr);
  void create_boolector_context(void);
  bool convert_identifier(const std::string &identifier, const typet &type, BtorExp* &bv);
  void print_data_types(BtorExp* operand0, BtorExp* operand1);
  std::string extract_magnitude(std::string v, unsigned width);
  std::string extract_fraction(std::string v, unsigned width);
  int literal_flag;
  unsigned int number_variables_boolector, set_to_counter, variable_number;
  Btor *boolector_ctx;
  FILE *btorFile, *smtFile;
  std::string filename;

  struct identifiert
  {
    typet type;
    exprt value;

    identifiert()
    {
      type.make_nil();
      value.make_nil();
    }
  };

  typedef hash_map_cont<irep_idt, identifiert, irep_id_hash>
    identifier_mapt;

  identifier_mapt identifier_map;
};

#endif
