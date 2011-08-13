/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BV_POINTERS_H
#define CPROVER_BV_POINTERS_H

#include <hash_cont.h>

#include "boolbv.h"
#include "pointer_logic.h"

class bv_pointerst:public boolbvt
{
public:
  bv_pointerst(propt &_prop);

  virtual void post_process();

protected:
  pointer_logict pointer_logic;

  typedef boolbvt SUB;

  unsigned addr_bits, offset_bits, bits;
  
  void encode(unsigned addr, bvt &bv);
  
  virtual void convert_pointer_type(const exprt &expr, bvt &bv);
  
  virtual void add_addr(const exprt &expr, bvt &bv);
  
  // overloading
  virtual literalt convert_rest(const exprt &expr);
  
  virtual void convert_bitvector(const exprt &expr, bvt &bv); // no cache

  virtual exprt bv_get_rec(
    const bvt &bv,
    const std::vector<bool> &unknown,
    unsigned offset,
    const typet &type) const;

  void convert_address_of_rec(
    const exprt &expr,
    bvt &bv);
    
  void increase_offset(bvt &bv, const mp_integer &x);
  void increase_offset(bvt &bv, const mp_integer &factor, const exprt &index);
  void increase_offset(bvt &bv, const mp_integer &factor, const bvt &index_bv);
  
  struct is_dynamic_objectt
  {
    bvt bv;
    literalt l;
  };
  
  typedef std::list<is_dynamic_objectt> is_dynamic_object_listt;
  is_dynamic_object_listt is_dynamic_object_list;  
  
  void do_is_dynamic_object(const is_dynamic_objectt &is_dynamic_object);
  
  static bool is_ptr(const typet &type)
  {
    return type.is_pointer() || type.is_reference();
  }
};

#endif
