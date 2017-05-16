/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_EXPR_H
#define CPROVER_EXPR_H

#include <iostream>
#include <util/location.h>
#include <util/type.h>

#define forall_operands(it, expr) \
  if((expr).has_operands()) \
    for(exprt::operandst::const_iterator it=(expr).operands().begin(); \
        it!=(expr).operands().end(); it++)

#define Forall_operands(it, expr) \
  if((expr).has_operands()) \
    for(exprt::operandst::iterator it=(expr).operands().begin(); \
        it!=(expr).operands().end(); it++)

#define forall_expr(it, expr) \
  for(exprt::operandst::const_iterator it=(expr).begin(); \
      it!=(expr).end(); it++)

#define Forall_expr(it, expr) \
  for(exprt::operandst::iterator it=(expr).begin(); \
      it!=(expr).end(); it++)
      
#define forall_expr_list(it, expr) \
  for(expr_listt::const_iterator it=(expr).begin(); \
      it!=(expr).end(); it++)

#define Forall_expr_list(it, expr) \
  for(expr_listt::iterator it=(expr).begin(); \
      it!=(expr).end(); it++)
      
class exprt:public irept
{
public:
  #ifdef USE_LIST
  typedef std::list<exprt> operandst;
  #else
  typedef std::vector<exprt> operandst;
  #endif

  // constructors
  exprt() { }
  explicit exprt(const irep_idt &_id):irept(_id) { }
  exprt(const irep_idt &_id, const typet &_type):irept(_id) { type()=_type; }
 
  bool has_operands() const
  { return !find(o_operands).is_nil(); }

  operandst &operands()
  { return (operandst &)(add(o_operands).get_sub()); }
  
  const operandst &operands() const
  { return (const operandst &)(find(o_operands).get_sub()); }

  const irep_idt &value(void) const {
    return get(a_value);
  }

  void value(irep_idt val) {
    set(a_value, val);
  };

  exprt &op0()
  { return operands().front(); }

  exprt &op1()
  #ifdef USE_LIST
  { return *(++operands().begin()); }
  #else
  { return operands()[1]; }
  #endif
   
  exprt &op2()
  #ifdef USE_LIST
  { return *(++ ++operands().begin()); }
  #else
  { return operands()[2]; }
  #endif
   
  exprt &op3()
  #ifdef USE_LIST
  { return *(++ ++ ++operands().begin()); }
  #else
  { return operands()[3]; }
  #endif
   
  const exprt &op0() const
  { return operands().front(); }

  const exprt &op1() const
  #ifdef USE_LIST
  { return *(++operands().begin()); }
  #else
  { return operands()[1]; }
  #endif
  
  const exprt &op2() const
  #ifdef USE_LIST
  { return *(++ ++operands().begin()); }
  #else
  { return operands()[2]; }
  #endif
  
  const exprt &op3() const
  #ifdef USE_LIST
  { return *(++ ++ ++operands().begin()); }
  #else
  { return operands()[3]; }
  #endif
  
  void reserve_operands(unsigned n)
  #ifdef USE_LIST
  { }
  #else
  { operands().reserve(n) ; }
  #endif
   
  void move_to_operands(exprt &expr); // destroys expr
  void move_to_operands(exprt &e1, exprt &e2); // destroys e1, e2
  void move_to_operands(exprt &e1, exprt &e2, exprt &e3); // destroys e1, e2, e3
  void copy_to_operands(const exprt &expr); // does not destroy expr
  void copy_to_operands(const exprt &e1, const exprt &e2); // does not destroy expr
  void copy_to_operands(const exprt &e1, const exprt &e2, const exprt &e3); // does not destroy expr

  void make_typecast(const typet &_type);
  void make_not();
  
  void make_true();
  void make_false();
  void make_bool(bool value);
  void negate();
  bool sum(const exprt &expr);
  bool mul(const exprt &expr);
  bool subtract(const exprt &expr);
  
  bool is_constant() const;
  bool is_true() const;
  bool is_false() const;
  bool is_zero() const;
  bool is_one() const;
  bool is_boolean() const;
  
  friend bool operator<(const exprt &X, const exprt &Y);
  
  const locationt &find_location() const;

  const locationt &location() const
  {
    return static_cast<const locationt &>(cmt_location());
  }

  locationt &location()
  {
    return static_cast<locationt &>(add(o_location));
  }
  
  exprt &add_expr(const std::string &name)
  {
    return static_cast<exprt &>(add(name));
  }

  const exprt &find_expr(const std::string &name) const
  {
    return static_cast<const exprt &>(find(name));
  }

  // Actual expression nodes
  static irep_idt trans;
  static irep_idt symbol;
  static irep_idt plus;
  static irep_idt minus;
  static irep_idt mult;
  static irep_idt div;
  static irep_idt mod;
  static irep_idt equality;
  static irep_idt notequal;
  static irep_idt index;
  static irep_idt arrayof;
  static irep_idt objdesc;
  static irep_idt dynobj;
  static irep_idt typecast;
  static irep_idt implies;
  static irep_idt i_and;
  static irep_idt i_xor;
  static irep_idt i_or;
  static irep_idt i_not;
  static irep_idt addrof;
  static irep_idt deref;
  static irep_idt i_if;
  static irep_idt with;
  static irep_idt member;
  static irep_idt isnan;
  static irep_idt ieee_floateq;
  static irep_idt i_type;
  static irep_idt constant;
  static irep_idt i_true;
  static irep_idt i_false;
  static irep_idt i_lt;
  static irep_idt i_gt;
  static irep_idt i_le;
  static irep_idt i_ge;
  static irep_idt i_bitand;
  static irep_idt i_bitor;
  static irep_idt i_bitxor;
  static irep_idt i_bitnand;
  static irep_idt i_bitnor;
  static irep_idt i_bitnxor;
  static irep_idt i_bitnot;
  static irep_idt i_ashr;
  static irep_idt i_lshr;
  static irep_idt i_shl;
  static irep_idt abs;
  static irep_idt argument;

  // Expression attributes
  static irep_idt a_value;

  // Other foo
protected:
  static irep_idt o_operands;
  static irep_idt o_location;
};

typedef std::list<exprt> expr_listt;

#endif
