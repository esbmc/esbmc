#ifndef CPROVER_POINTER_LOGIC_H
#define CPROVER_POINTER_LOGIC_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/mp_arith.h>
#include <util/numbering.h>

class pointer_logict
{
public:
  // this numbers the objects
  typedef std::unordered_map<expr2tc, unsigned int, irep2_hash> objectst;
  objectst objects;
  typedef std::vector<expr2tc> obj_lookupt;
  obj_lookupt lookup;
  unsigned int obj_num_offset;

  struct pointert
  {
    unsigned object;
    BigInt offset;

    pointert() = default;

    pointert(unsigned _obj, const BigInt &_off) : object(_obj), offset(_off)
    {
    }
  };

  // converts an (object,offset) pair to an expression
  expr2tc pointer_expr(const pointert &pointer, const type2tc &type) const;

  // converts an (object,0) pair to an expression
  expr2tc pointer_expr(unsigned object, const type2tc &type) const;

  ~pointer_logict() = default;
  pointer_logict();

  unsigned add_object(const expr2tc &expr);

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

  unsigned get_free_obj_num()
  {
    return objects.size() - 1 + ++obj_num_offset;
  }

protected:
  unsigned null_object, invalid_object;

  expr2tc pointer_expr(const BigInt &offset, const expr2t &object) const;

  expr2tc object_rec(
    const BigInt &offset,
    const type2tc &pointer_type,
    const expr2tc &src) const;
};

#endif
