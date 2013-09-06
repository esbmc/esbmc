/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_VALUE_SET_H
#define CPROVER_POINTER_ANALYSIS_VALUE_SET_H

#include <irep2.h>

#include <set>

#include <mp_arith.h>
#include <namespace.h>
#include <reference_counting.h>

#include "object_numbering.h"
#include "value_sets.h"

/** Code for tracking "value sets" across assignments in ESBMC.
 *
 *  The values in a value set are /references/ to /objects/, and additional data
 *  about the reference itself. You can consider any level 1 renamed variable
 *  to be such an object; and a set of them can identify the set of things that
 *  a pointer points at.
 *
 *  The way ESBMC uses this is by keeping a mapping of (l1) pointer variables,
 *  and the set of data objects that they can point at. This mapping is then
 *  updated during the execution of the program, during which the assignments
 *  made are interpreted by the code in value_sett: pointer variable assignments
 *  are detected, their right hand sides interpreted to determine what objects
 *  may be referred to, and the mapping updated to show that the left hand side
 *  variable may point at those objects. Phi nodes are handled by merging the
 *  mappings of each pointer variable.
 *
 *  As well as keeping track of pointer variables and the set of things they can
 *  point at during symbolic execution, a static analysis also uses value_sett
 *  to track pointer variable assignments. This is on an even more abstract
 *  level, to compute a points-to analysis of all the code under test.
 *  That is, the outcome is a set of what all (l0) variables /might/ point at,
 *  This is done using exactly the same mapping/interpretation code, as well
 *  as logic (elsewhere) for computing a fixedpoint, and some black magic that
 *  attempts to statically track dynamically allocated memory.
 *
 *  The only data element stored is a map from l1 variable names (as strings)
 *  to a record of what objects are stored. Data objects are numbered, with the
 *  mapping for that stored in a global variable, value_sett::object_numbering,
 *  which will explode into multithreaded death cakes in the future. The primary
 *  interfaces to the value_sett object itself are the 'assign' method (for
 *  interpreting a variable assignment) and the get_value_set method, that takes
 *  a variable and returns the set of things it might point at.
 */

class value_sett
{
public:
  /** Primary constructor. Does approximately nothing non-standard. */
  value_sett():location_number(0)
  {
  }

//*********************************** Types ************************************

  /** A type for a set of expressions */
  typedef std::set<expr2tc> expr_sett;

  /** Record for an object reference. Any reference to an object is stored as
   *  an objectt, as a map element in an object_mapt. The actual object that
   *  this refers to is determined by the /key/ of this objectt in the
   *  object_mapt map.
   *
   *  This class itself just stores additional information about that reference:
   *  how far into the object does the pointer reference point, if there's an
   *  offset. Alternately, if the offset is not known, due to nondeterminism or
   *  otherwise, then this records that the offset can't be determined
   *  statically and must be evaluated at solver time. */
  class objectt
  {
  public:
    objectt():offset_is_set(false)
    {
    }

    explicit objectt(const mp_integer &_offset):
      offset(_offset),
      offset_is_set(true)
    {
    }

    /** Record of the explicit offset into the object. Only valid when
     *  offset_is_set is true. */
    mp_integer offset;
    /** Whether or not the offset field of this objectt is valid; if this is
     *  true, then the reference has a fixed offset into the object, the value
     *  of which is in the offset field. If not, then the offset isn't
     *  statically known, and must be handled at solver time. */
    bool offset_is_set;
    bool offset_is_zero() const
    { return offset_is_set && offset.is_zero(); }
  };

  /** Datatype for a value set: stores a mapping between some integers and
   *  additional reference data in an objectt object. The integers are indexes
   *  into value_sett::object_numbering, which identifies the l1 variable
   *  being referred to.
   *
   *  This code commits the sin of extending an STL type. Bad. */
  class object_map_dt:public std::map<unsigned, objectt>
  {
  public:
    const static object_map_dt empty;
  };

  /** Reference counting wrapper around an object_map_dt. */
  typedef reference_counting<object_map_dt> object_mapt;

  /** Record for a particular value set: stores the identity of the variable
   *  that points at this set of objects, and the objects themselves (with
   *  associated offset data).
   */
  struct entryt
  {
    /** The map of objects -> their offset data. Any key/value pair in this
     *  map represents a object/offset-data (respectively) that this variable
     *  can point at. */
    object_mapt object_map;
    /** The L1 name of the pointer variable that's doing the pointing. */
    std::string identifier;
    /** Additional suffix data -- an L1 variable might actually contain several
     *  pointers. For example, an array of pointer, or a struct with multiple
     *  pointer members. This suffix uniquely distinguishes which pointer
     *  variable (within the l1 variable) this record is for. As an example,
     *  it might read '.ptr' to identify the ptr field of a struct. */
    std::string suffix;

    entryt()
    {
    }

    entryt(const std::string &_identifier, const std::string _suffix):
      identifier(_identifier),
      suffix(_suffix)
    {
    }
  };

  /** Type of the value-set containing structure. A hash map mapping variables
   *  to an entryt, storing the value set of objects a variable might point
   *  at. */
  typedef hash_map_cont<string_wrapper, entryt, string_wrap_hash> valuest;

//********************************** Methods ***********************************

  /** Convert an object map element to an expression. Formulates either an
   *  object_descriptor irep, or unknown / invalid expr's as appropriate. */
  expr2tc to_expr(object_map_dt::const_iterator it) const;

  /** Insert an object record element into an object map.
   *  @param dest The map to insert this record into.
   *  @param it Iterator of existing object record to insert into dest. */
  void set(object_mapt &dest, object_map_dt::const_iterator it) const
  {
    dest.write()[it->first]=it->second;
  }

  bool insert(object_mapt &dest, object_map_dt::const_iterator it) const
  {
    return insert(dest, it->first, it->second);
  }

  bool insert(object_mapt &dest, const expr2tc &src) const
  {
    return insert(dest, object_numbering.number(src), objectt());
  }

  bool insert(object_mapt &dest, const expr2tc &src, const mp_integer &offset) const
  {
    return insert(dest, object_numbering.number(src), objectt(offset));
  }

  /** Insert an object record into the given object map. This method has
   *  various overloaded instances, that all descend to this particular method.
   *  The essential elements are a) an object map, b) an l1 data object or
   *  the index number (in value_sett::object_numbering) that identifies
   *  it, and c) the offset data for this record.
   *
   *  Rather than just adding this pointer record to the object map, this
   *  method attempts to merge the data in. That is, if the variable might
   *  already point at the same object, attempt to merge the two offset
   *  records. If that fails, then record the offset as being nondeterministic,
   *  and let the SMT solver work it out.
   *
   *  @param dest The object map to insert this record into.
   *  @param n The identifier for the object being referrred to, as indexed by
   *         the value_set::object_numbering mapping.
   *  @param object The offset data for the pointer record being inserted.
   */
  bool insert(object_mapt &dest, unsigned n, const objectt &object) const
  {
    if(dest.read().find(n)==dest.read().end())
    {
      // new
      dest.write()[n]=object;
      return true;
    }
    else
    {
      objectt &old=dest.write()[n];

      if(old.offset_is_set && object.offset_is_set)
      {
        if(old.offset==object.offset)
          return false;
        else
        {
          old.offset_is_set=false;
          return true;
        }
      }
      else if(!old.offset_is_set)
        return false;
      else
      {
        old.offset_is_set=false;
        return true;
      }
    }
  }

  bool insert(object_mapt &dest, const expr2tc &expr, const objectt &object) const
  {
    return insert(dest, object_numbering.number(expr), object);
  }

  /** Remove the given pointer value set from the map.
   *  @param name The name of the variable, including suffix, to erase.
   *  @return True when the erase succeeds, false otherwise. */
  bool erase(const std::string &name)
  {
    return (values.erase(string_wrapper(name)) == 1);
  }

  /**  */
  void get_value_set(
    const expr2tc &expr,
    value_setst::valuest &dest,
    const namespacet &ns) const;

  void clear()
  {
    values.clear();
  }

  void add_var(const std::string &id, const std::string &suffix)
  {
    get_entry(id, suffix);
  }

  void add_var(const entryt &e)
  {
    get_entry(e.identifier, e.suffix);
  }

  void del_var(const std::string &id, const std::string &suffix)
  {
    std::string index = id2string(id) + suffix;
    values.erase(index);
  }

  entryt &get_entry(const std::string &id, const std::string &suffix)
  {
    return get_entry(entryt(id, suffix));
  }

  entryt &get_entry(const entryt &e)
  {
    std::string index=id2string(e.identifier)+e.suffix;

    std::pair<valuest::iterator, bool> r=
      values.insert(std::pair<string_wrapper, entryt>
                             (string_wrapper(index), e));

    return r.first->second;
  }

  void add_vars(const std::list<entryt> &vars)
  {
    for(std::list<entryt>::const_iterator
        it=vars.begin();
        it!=vars.end();
        it++)
      add_var(*it);
  }

  void output(
    const namespacet &ns,
    std::ostream &out) const;

  void dump(const namespacet &ns) const;

  // true = added s.th. new
  bool make_union(object_mapt &dest, const object_mapt &src) const;

  // true = added s.th. new
  bool make_union(const valuest &new_values, bool keepnew=false);

  // true = added s.th. new
  bool make_union(const value_sett &new_values, bool keepnew=false)
  {
    return make_union(new_values.values, keepnew);
  }

  void apply_code(
    const expr2tc &code,
    const namespacet &ns);

  void assign(
    const expr2tc &lhs,
    const expr2tc &rhs,
    const namespacet &ns,
    bool add_to_sets=false);

  void do_function_call(
    const irep_idt &function,
    const std::vector<expr2tc> &arguments,
    const namespacet &ns);

  // edge back to call site
  void do_end_function(
    const expr2tc &lhs,
    const namespacet &ns);

  void get_reference_set(
    const expr2tc &expr,
    value_setst::valuest &dest,
    const namespacet &ns) const;

protected:
  void get_value_set_rec(
    const expr2tc &expr,
    object_mapt &dest,
    const std::string &suffix,
    const type2tc &original_type,
    const namespacet &ns) const;

  void get_value_set(
    const expr2tc &expr,
    object_mapt &dest,
    const namespacet &ns) const;

  void get_reference_set(
    const expr2tc &expr,
    object_mapt &dest,
    const namespacet &ns) const
  {
    get_reference_set_rec(expr, dest, ns);
  }

  void get_reference_set_rec(
    const expr2tc &expr,
    object_mapt &dest,
    const namespacet &ns) const;

  void assign_rec(
    const expr2tc &lhs,
    const object_mapt &values_rhs,
    const std::string &suffix,
    const namespacet &ns,
    bool add_to_sets);

  void do_free(
    const expr2tc &op,
    const namespacet &ns);

  expr2tc make_member(
    const expr2tc &src,
    const irep_idt &component_name,
    const namespacet &ns);

public:
//********************************** Members ***********************************
  unsigned location_number;
  static object_numberingt object_numbering;

  valuest values;
};

#endif
