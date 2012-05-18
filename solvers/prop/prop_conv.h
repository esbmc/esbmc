/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

//
// convert expression to boolean formula
//

#ifndef CPROVER_PROP_CONV_H
#define CPROVER_PROP_CONV_H

#include <string>
#include <list>

#include <irep2.h>
#include <config.h>
#include <hash_cont.h>
#include <message.h>
#include <threeval.h>

#include "literal.h"

class prop_convt : public messaget
{
public:
  explicit prop_convt() { }
  virtual ~prop_convt() { }

  typedef enum { P_SATISFIABLE, P_UNSATISFIABLE, P_ERROR, P_SMTLIB } resultt;

  // overloading
  virtual void set_to(const expr2tc &expr, bool value) = 0;
  virtual resultt dec_solve() = 0;

  virtual literalt convert(const expr2tc &expr);

  // get a value from counterexample if not valid
  virtual expr2tc get(const expr2tc &expr) const = 0;
  
  virtual void clear_cache()
  {
    cache.clear();
  }
  
  // Literal manipulation routines inhereted from propt.
  virtual literalt land(literalt a, literalt b)=0;
  virtual literalt lor(literalt a, literalt b)=0;
  virtual literalt land(const bvt &bv)=0;
  virtual literalt lor(const bvt &bv)=0;
  virtual literalt lnot(literalt a)=0;
  virtual literalt limplies(literalt a, literalt b)=0;
  virtual void set_equal(literalt a, literalt b);

  virtual void l_set_to(literalt a, bool value)
  {
    set_equal(a, const_literal(value));
  }

  // constraints
  virtual void lcnf(const bvt &bv)=0;

  // variables
  virtual literalt new_variable()=0;
  virtual unsigned no_variables() const=0;

  // solving
  virtual const std::string solver_text()=0;

  // satisfying assignment
  virtual tvt l_get(literalt a) const=0;

protected:
  virtual literalt convert_expr(const expr2tc &expr) = 0;
  
  // symbols
  typedef std::map<irep_idt, literalt> symbolst;
  symbolst symbols;

  // cache
  typedef hash_map_cont<expr2tc, literalt, irep2_hash> cachet;
  cachet cache;
  
  virtual void ignoring(const expr2tc &expr);

public:
  const cachet &get_cache() const { return cache; }

  virtual void convert_smt_type(const type2t &type, void *&arg);
  virtual void convert_smt_type(const bool_type2t &type, void *&arg) = 0;
  virtual void convert_smt_type(const unsignedbv_type2t &type, void *&arg) = 0;
  virtual void convert_smt_type(const signedbv_type2t &type, void *&arg) = 0;
  virtual void convert_smt_type(const array_type2t &type, void *&arg) = 0;
  virtual void convert_smt_type(const pointer_type2t &type, void *&arg) = 0;
  virtual void convert_smt_type(const struct_type2t &type, void *&arg) =0;
  virtual void convert_smt_type(const union_type2t &type, void *&arg) =0;
  virtual void convert_smt_type(const fixedbv_type2t &type, void *&arg) =0;

  virtual void convert_smt_expr(const expr2t &expr, void *&arg);
  virtual void convert_smt_expr(const symbol2t &sym, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_int2t &sym, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_fixedbv2t &bv, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_bool2t &sym, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_struct2t &strt, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_union2t &strct, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_array2t &array, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_array_of2t &array, void *&arg) = 0;
  virtual void convert_smt_expr(const constant_string2t &str, void *&arg) = 0;
  virtual void convert_smt_expr(const if2t &ifirep, void *&arg) = 0;
  virtual void convert_smt_expr(const equality2t &equality, void *&arg) = 0;
  virtual void convert_smt_expr(const notequal2t &notequal, void *&arg) = 0;
  virtual void convert_smt_expr(const lessthan2t &lessthan, void *&arg) = 0;
  virtual void convert_smt_expr(const greaterthan2t &greaterthan, void *&arg) = 0;
  virtual void convert_smt_expr(const lessthanequal2t &le, void *&arg) = 0;
  virtual void convert_smt_expr(const greaterthanequal2t &le, void *&arg) = 0;
  virtual void convert_smt_expr(const not2t &notval, void *&arg) = 0;
  virtual void convert_smt_expr(const and2t &andval, void *&arg) = 0;
  virtual void convert_smt_expr(const or2t &orval, void *&arg) = 0;
  virtual void convert_smt_expr(const xor2t &xorval, void *&arg) = 0;
  virtual void convert_smt_expr(const implies2t &implies, void *&arg) = 0;
  virtual void convert_smt_expr(const bitand2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const bitor2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const bitxor2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const bitnand2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const bitnor2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const bitnxor2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const bitnot2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const lshr2t &bitval, void *&arg) = 0;
  virtual void convert_smt_expr(const neg2t &neg, void *&arg) = 0;
  virtual void convert_smt_expr(const abs2t &abs, void *&arg) = 0;
  virtual void convert_smt_expr(const add2t &add, void *&arg) = 0;
  virtual void convert_smt_expr(const sub2t &sub, void *&arg) = 0;
  virtual void convert_smt_expr(const mul2t &mul, void *&arg) = 0;
  virtual void convert_smt_expr(const div2t &mul, void *&arg) = 0;
  virtual void convert_smt_expr(const modulus2t &mod, void *&arg) = 0;
  virtual void convert_smt_expr(const shl2t &shl, void *&arg) = 0;
  virtual void convert_smt_expr(const ashr2t &ashr, void *&arg) = 0;
  virtual void convert_smt_expr(const same_object2t &same, void *&arg) = 0;
  virtual void convert_smt_expr(const pointer_offset2t &offs, void *&arg) = 0;
  virtual void convert_smt_expr(const pointer_object2t &obj, void *&arg) = 0;
  virtual void convert_smt_expr(const address_of2t &obj, void *&arg) = 0;
  virtual void convert_smt_expr(const byte_extract2t &data, void *&arg) = 0;
  virtual void convert_smt_expr(const byte_update2t &data, void *&arg) = 0;
  virtual void convert_smt_expr(const with2t &with, void *&arg) = 0;
  virtual void convert_smt_expr(const member2t &member, void *&arg) = 0;
  virtual void convert_smt_expr(const typecast2t &cast, void *&arg) = 0;
  virtual void convert_smt_expr(const index2t &index, void *&arg) = 0;
  virtual void convert_smt_expr(const zero_string2t &zstr, void *&arg) = 0;
  virtual void convert_smt_expr(const zero_length_string2t &s, void *&arg) = 0;
  virtual void convert_smt_expr(const isnan2t &isnan, void *&arg) = 0;
  virtual void convert_smt_expr(const overflow2t &overflow, void *&arg) = 0;
  virtual void convert_smt_expr(const overflow_cast2t &ocast, void *&arg) = 0;
  virtual void convert_smt_expr(const overflow_neg2t &neg, void *&arg) = 0;
  virtual void convert_smt_expr(const buffer_size2t &buf, void *&arg) = 0;
};

#endif
