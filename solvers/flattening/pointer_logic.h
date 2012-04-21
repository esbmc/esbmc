/*******************************************************************\

Module: Pointer Logic

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_LOGIC_H
#define CPROVER_POINTER_LOGIC_H

#include <mp_arith.h>
#include <hash_cont.h>
#include <expr.h>
#include <numbering.h>

class pointer_logict
{
public:
  // this numbers the objects
  typedef hash_numbering<exprt, irep_hash> objectst;
  objectst objects;

  struct pointert
  {
    unsigned object;
    mp_integer offset;
    
    pointert()
    {
    }
    
    pointert(unsigned _obj, mp_integer _off):object(_obj), offset(_off)
    {
    }
  };
  
  // converts an (object,offset) pair to an expression
  exprt pointer_expr(
    const pointert &pointer,
    const typet &type) const;

  // converts an (object,0) pair to an expression
  exprt pointer_expr(
    unsigned object,
    const typet &type) const;
    
  ~pointer_logict();
  pointer_logict();
  
  unsigned add_object(const exprt &expr);

  // number of NULL object  
  unsigned get_null_object() const
  {
    return null_object;
  }

  // number of INVALID object  
  unsigned get_invalid_object() const
  {
    return invalid_object;
  }
  
protected:
  unsigned null_object, invalid_object;  

  exprt pointer_expr(
    const mp_integer &offset,
    const exprt &object) const;

  exprt object_rec(
    const mp_integer &offset,
    const typet &pointer_type,
    const exprt &src) const;
};

#endif
