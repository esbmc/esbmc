/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SIMPLIFY_EXPR_CLASS_H
#define CPROVER_SIMPLIFY_EXPR_CLASS_H

#include <map>
#include <set>
#include <util/expr.h>
#include <util/hash_cont.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/threeval.h>

#define forall_value_list(it, value_list) \
  for(simplify_exprt::value_listt::const_iterator it=(value_list).begin(); \
      it!=(value_list).end(); it++)

class simplify_exprt
{
public:
  bool do_simplify_if;

  bool simplify_typecast(exprt &expr);
  bool simplify_concatenation(exprt &expr);
  bool simplify_multiplication(exprt &expr);
  bool simplify_division(exprt &expr);
  bool simplify_modulo(exprt &expr);
  bool simplify_addition_substraction(exprt &expr);
  bool simplify_shifts(exprt &expr);
  bool simplify_bitwise(exprt &expr);
  bool simplify_if_implies(exprt &expr, const exprt &cond, bool truth, bool &new_truth);
  bool simplify_if_recursive(exprt &expr, const exprt &cond, bool truth);
  bool simplify_if_conj(exprt &expr, const exprt &cond);
  bool simplify_if_disj(exprt &expr, const exprt &cond);
  bool simplify_if_branch(exprt &trueexpr, exprt &falseexpr, const exprt &cond);
  bool simplify_if_cond(exprt &expr);
  bool simplify_if(exprt &expr);
  bool simplify_switch(exprt &expr);
  bool simplify_bitnot(exprt &expr);
  bool simplify_not(exprt &expr);
  bool simplify_boolean(exprt &expr);
  bool simplify_inequality(exprt &expr);
  bool simplify_inequality_constant(exprt &expr);
  bool simplify_inequality_not_constant(exprt &expr);
  bool simplify_relation(exprt &expr);
  bool simplify_ieee_float_relation(exprt &expr);
  bool simplify_with(exprt &expr);
  bool simplify_index(index_exprt &expr);
  bool simplify_member(member_exprt &expr);
  bool simplify_pointer_object(exprt &expr);
  bool simplify_dynamic_size(exprt &expr);
  bool simplify_is_dynamic_object(exprt &expr);
  bool simplify_same_object(exprt &expr);
  bool simplify_valid_object(exprt &expr);
  bool simplify_object(exprt &expr);
  static tvt objects_equal(const exprt &a, const exprt &b);
  static tvt objects_equal_address_of(const exprt &a, const exprt &b);
  bool sort_and_join(exprt &expr);
  bool simplify_unary_minus(exprt &expr);
  bool simplify_dereference(exprt &expr);
  bool simplify_address_of(exprt &expr);
  bool simplify_pointer_offset(exprt &expr);
  bool eliminate_common_addends(exprt &op0, exprt &op1);
  exprt pointer_offset(const exprt &expr, const typet &type);

  virtual bool simplify_node(exprt &expr);
  virtual bool simplify_rec(exprt &expr);

  virtual bool simplify(exprt &expr)
  {
    return simplify_rec(expr);
  }

  virtual ~simplify_exprt() = default;

  typedef std::set<mp_integer> value_listt;
  bool get_values(const exprt &expr, value_listt &value_list);

  simplify_exprt():do_simplify_if(true)
  {
  }

  static bool is_bitvector_type(const typet &type)
  {
    return type.id()=="unsignedbv" || type.id()=="signedbv" || type.id()=="bv";
  }
};

bool sort_and_join(const std::string &id, const std::string &type_id);

#endif
