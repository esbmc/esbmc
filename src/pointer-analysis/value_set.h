/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_VALUE_SET_H
#define CPROVER_POINTER_ANALYSIS_VALUE_SET_H

#include <pointer-analysis/object_numbering.h>
#include <pointer-analysis/value_sets.h>
#include <set>
#include <util/irep2.h>
#include <util/mp_arith.h>
#include <util/namespace.h>
#include <util/reference_counting.h>
#include <util/type_byte_size.h>

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
 *  (which will explode into multithreaded death cakes in the future). The
 *  primary interfaces to the value_sett object itself are the 'assign' method
 *  (for interpreting a variable assignment) and the get_value_set method, that
 *  takes a variable and returns the set of things it might point at.
 */

class value_sett
{
public:
  /** Primary constructor. Does approximately nothing non-standard. */
  value_sett(const namespacet &_ns):location_number(0), ns(_ns),
    xchg_name("value_sett::__ESBMC_xchg_ptr"), xchg_num(0)
  {
  }

  value_sett(const value_sett &ref) :
    location_number(ref.location_number),
    values(ref.values),
    ns(ref.ns),
    xchg_name("value_sett::__ESBMC_xchg_ptr"),
    xchg_num(0)
  {
  }

  value_sett &operator=(const value_sett &ref)
  {
    location_number = ref.location_number;
    values = ref.values;
    xchg_name = ref.xchg_name;
    xchg_num = ref.xchg_num;
    // No need to copy ns, it should be the same in all contexts.
    return *this;
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
    objectt() : offset(0), offset_is_set(true), offset_alignment(0) { }

    objectt(bool offset_set, unsigned int operand)
    {
      if (offset_set) {
        offset_is_set = true;
        offset = mp_integer(operand);
        offset_alignment = 1;
      } else {
        offset_is_set = false;
        offset_alignment = operand;
        assert(offset_alignment != 0);
      }
    }

    explicit objectt(bool offset_set __attribute__((unused)),
        const mp_integer &_offset):
      offset(_offset),
      offset_is_set(true)
    {
      assert(offset_set);
      offset_alignment = 1;
      // offset_set = offset_set;
    }

    /** Record of the explicit offset into the object. Only valid when
     *  offset_is_set is true. */
    mp_integer offset;
    /** Whether or not the offset field of this objectt is valid; if this is
     *  true, then the reference has a fixed offset into the object, the value
     *  of which is in the offset field. If not, then the offset isn't
     *  statically known, and must be handled at solver time. */
    bool offset_is_set;
    /** Least alignment of the offset. When offset_is_set is false, we state
     *  what we think the alignment of this pointer is. This becomes massively
     *  useful if we point at an array, but can know that the pointer is aligned
     *  to the array element edges.
     *  Units are bytes. Zero means N/A. */
    unsigned int offset_alignment;
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
    object_map_dt() = default;
    const static object_map_dt empty;
  };

  /** Reference counting wrapper around an object_map_dt. */
  typedef reference_counting<object_map_dt> object_mapt;

  /** Record for a particular value set: stores the identity of the variable
   *  that points at this set of objects, and the objects themselves (with
   *  associated offset data).
   *
   *  The canonical name of this entry, stored in value_set::values, is the
   *  'identifier' field concatonated with the 'suffix' field (see docs for
   *  suffix).
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
     *  it might read '.ptr' to identify the ptr field of a struct. It might
     *  also be '[]' if this is the value set of an array of pointers: we don't
     *  track each individual element, only the array of them. */
    std::string suffix;

    entryt() = default;

    entryt(std::string _identifier, const std::string& _suffix):
      identifier(std::move(_identifier)),
      suffix(_suffix)
    {
    }
  };

  /** Type of the value-set containing structure. A hash map mapping variables
   *  to an entryt, storing the value set of objects a variable might point
   *  at. */
  typedef hash_map_cont<string_wrapper, entryt, string_wrap_hash> valuest;

  /** Get the natural alignment unit of a reference to e. I don't know a more
   *  appropriate term, but if we were to have an offset into e, then what is
   *  the greatest alignment guarentee that would make sense? i.e., an offset
   *  of 404 into an array of integers gives an alignment guarentee of 4 bytes,
   *  not 404.
   *
   *  For arrays, this is the element size.
   *  For structs, I imagine it's the machine word size (?). Depends on padding.
   *  For integers / other things, it's the machine / word size (?).
   */
  inline unsigned int get_natural_alignment(const expr2tc &e) const
  {
    const type2tc &t = e->type;

    // Null objects are allowed to have symbol types. What alignment to give?
    // Pick 8 bytes, as that's a) word aligned, b) double/uint64_t aligned.
    if (is_null_object2t(e))
      return 8;

    assert(!is_symbol_type(t));
    if (is_array_type(t)) {
      const array_type2t &arr = to_array_type(t);
      return type_byte_size_default(arr.subtype, 8).to_ulong();
    } else {
      return 8;
    }
  }

  inline unsigned int offset2align(const expr2tc &e, const mp_integer &m) const
  {
    unsigned int nat_align = get_natural_alignment(e);
    if (m == 0) {
      return nat_align;
    } else if ((m % nat_align) == 0) {
      return nat_align;
    } else {
      // What's the least alignment available?
      unsigned int max_align = 8;
      do {
        // Repeatedly decrease the word size by powers of two, and test to see
        // whether the offset meets that alignment. This will always succeed
        // and exit the loop when the alignment reaches 1.
        if ((m % max_align) == 0)
          return max_align;
        max_align /= 2;
      } while (true);
    }
  }

  /** Convert an object map element to an expression. Formulates either an
   *  object_descriptor irep, or unknown / invalid expr's as appropriate. */
  expr2tc to_expr(object_map_dt::const_iterator it) const;

  /** Insert an object record element into an object map.
   *  @param dest The map to insert this record into.
   *  @param it Iterator of existing object record to insert into dest. */
  void set(object_mapt &dest, object_map_dt::const_iterator it) const
  {
    // Fetch/insert iterator
    std::pair<object_map_dt::iterator,bool> res =
      dest.write().insert(object_map_dt::value_type(it->first, it->second));

    // If element already existed, overwrite.
    if (res.second)
      res.first->second = it->second;
  }

  bool insert(object_mapt &dest, object_map_dt::const_iterator it) const
  {
    return insert(dest, it->first, it->second);
  }

  bool insert(object_mapt &dest, const expr2tc &src, const mp_integer &offset) const
  {
    return insert(dest, object_numbering.number(src), objectt(true, offset));
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
    object_map_dt::const_iterator it = dest.read().find(n);
    if (it == dest.read().end())
    {
      // new
      dest.write().insert(object_map_dt::value_type(n, object));
      return true;
    }
    else
    {
      object_map_dt::iterator it2 = dest.write().find(n);
      objectt &old = it2->second;
      const expr2tc &expr_obj = object_numbering[n];

      if(old.offset_is_set && object.offset_is_set)
      {
        if(old.offset==object.offset)
          return false;
        else
        {
          // Merge the tracking for two offsets; take the minimum alignment
          // guarenteed by them.
          unsigned long old_align = offset2align(expr_obj, old.offset);
          unsigned long new_align = offset2align(expr_obj, object.offset);
          old.offset_is_set = false;
          old.offset_alignment = std::min(old_align, new_align);
          return true;
        }
      } else if(!old.offset_is_set) {
        unsigned int oldalign = old.offset_alignment;
        if (!object.offset_is_set) {
          // Both object offsets not set; update alignment to minimum of the two
          old.offset_alignment =
            std::min(old.offset_alignment, object.offset_alignment);
          return !(old.offset_alignment == oldalign);
        } else {
          // Old offset unset; new offset set. Compute the alignment of the
          // new object's offset, and take the minimum of that and the old
          // alignment.
          unsigned int new_alignment = offset2align(expr_obj, object.offset);
          old.offset_alignment = std::min(old.offset_alignment, new_alignment);
          return !(old.offset_alignment == oldalign);
        }
      }
      else
      {
        // Old offset alignment is set; new isn't.
        unsigned int old_align = offset2align(expr_obj, old.offset);
        old.offset_alignment = std::min(old_align, object.offset_alignment);
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

  /** Get the set of things that an expression might point at. Interprets the
   *  given expression, making note of all the pointer variables used by it,
   *  and create a value set of what this expression points at. For example,
   *  the expression:
   *    (foo + 2)
   *  would return the set of all the things 'foo' points at, with the offset
   *  records updated to reflect the offset added by the expression. (Although
   *  not for records where the offset is nondeterministic.
   *
   *  To be clear: the value set returned reflects the values that the entire
   *  expression may evaluate to. That means if you have
   *    ((nondet_bool()) ? foo : bar)
   *  then you'll get everything both foo and bar point at.
   *
   *  This method also looks up values through dereferences and addresses-of.
   *
   *  @param expr The expression to evaluate and determine the set of values
   *         it points to.
   *  @param dest A list to store pointed-at object expressions into.
   *  */
  void get_value_set(const expr2tc &expr, value_setst::valuest &dest) const;

  /** Clear all value records from this value set. */
  void clear()
  {
    values.clear();
  }

  /** Add a value set for the given variable name and suffix. No effect if the
   *  given record already exists. */
  void add_var(const std::string &id, const std::string &suffix)
  {
    get_entry(id, suffix);
  }

  void add_var(const entryt &e)
  {
    get_entry(e.identifier, e.suffix);
  }

  /** Delete the value set for the given variable name and suffix. */
  void del_var(const std::string &id, const std::string &suffix)
  {
    std::string index = id2string(id) + suffix;
    values.erase(index);
  }

  /** Look up the value set for the given variable name and suffix. */
  entryt &get_entry(const std::string &id, const std::string &suffix)
  {
    return get_entry(entryt(id, suffix));
  }

  /** Look upt he value set for the variable name and suffix stored in the
   *  given entryt. */
  entryt &get_entry(const entryt &e)
  {
    std::string index=id2string(e.identifier)+e.suffix;

    std::pair<valuest::iterator, bool> r=
      values.insert(std::pair<string_wrapper, entryt>
                             (string_wrapper(index), e));

    return r.first->second;
  }

  /** Add a value set for each variable in the given list. */
  void add_vars(const std::list<entryt> &vars)
  {
    for(const auto & var : vars)
      add_var(var);
  }

  /** Dump the value set's textual representation to the given iostream.
   *  @param out Output stream to write the textual representation too. */
  void output(std::ostream &out) const;

  /** Write a textual representation of the value set to stderr. */
  void dump() const;

  /** Join the two given object maps. Takes all the pointer records from src
   *  and stores them into the dest object map.
   *  @param dest Destination object map to join records into.
   *  @param src Object map to merge into dest.
   *  @return True when dest has been modified. */
  bool make_union(object_mapt &dest, const object_mapt &src) const;

  /** Given another value set tracking object's storage, read all value set
   *  records out and merge them into this object's.
   *  @param new_values Stored set of value sets to merge into this object.
   *  @param keepnew If true, add new pointer records in new_values into this
   *         object's tracking map; if not, discard them.
   *  @return True if a modification occurs. */
  bool make_union(const valuest &new_values, bool keepnew=false);

  bool make_union(const value_sett &new_values, bool keepnew=false)
  {
    return make_union(new_values.values, keepnew);
  }

  /** When using value_sett for static analysis, takes a code statement and
   *  sends any assignments contained within to the assign method.
   *  @param code The statement to interpret. */
  void apply_code(const expr2tc &code);

  /** Interpret an assignment, and update value sets to reflect it.
   *  @param lhs Assignment target expression.
   *  @param rhs Assignment expression, to be interpreted, and its pointer
   *             records assigned to lhs.
   *  @param add_to_sets If true, merge the pointer set from rhs into the
   *         pointer set for lhs. Otherwise, overwrite it. Used for the static
   *         analysis. */
  void assign(
    const expr2tc &lhs,
    const expr2tc &rhs,
    bool add_to_sets=false);

  /** Interpret a function call during static analysis. Looks up the given
   *  function, and simulates the assignment of all the arguments to the
   *  argument variables in the target function (for pointer tracking).
   *  @param function Symbol of the function to bind arguments into.
   *  @param arguments Vector of argument expressions, will have their pointer
   *         tracking values merged into the corresponding argument variables
   *         in the target function. */
  void do_function_call(
    const symbolt &symbol,
    const std::vector<expr2tc> &arguments);

  /** During static analysis, simulate the return values assignment to the
   *  given lhs at the end of the function execution.
   *  @param lhs Variable to take the (pointer) values of the returned value. */
  void do_end_function(const expr2tc &lhs);

  /** Determine the set of variables that expr refers to. The difference between
   *  this and get_value_set, is that this accumulates the set of variables
   *  /used/ in the expression, not the values they might point at. The primary
   *  use is, when one uses the address-of operator, and the operand may be
   *  a set of things, determine that set of things. An example:
   *    baz = &a->foo.bar[0]
   *  What does baz point at? It depends on what a points at, and a series of
   *  other expressions that must be interpreted. This method performs said
   *  interpretation, storing the results in a value set.
   *
   *  @param expr The expression to evaluate the reference set for.
   *  @param dest The expression list to store the results into.
   */
  void get_reference_set(const expr2tc &expr, value_setst::valuest &dest) const;

protected:
  /** Recursive body of get_value_set.
   *  @param expr Expression to interpret and fetch value set for
   *  @param dest Destination object map to store pointed-at records in.
   *  @param suffix Cumulative suffix to attach to referred-to variables. See
   *         the documentation on @ref entryt for what the suffix means. As
   *         higher level expressions determine the suffix that a variable name
   *         gets, this must be passed down from higher levels (through this
   *         parameter).
   *  @param original_type Type of the top level expression. If any part of the
   *         interpreted expression isn't recognized, then an unknown2t expr is
   *         put in the value tracking set to represent the fact that
   *         interpretation failed, and it might point at something crazy. */
  void get_value_set_rec(
    const expr2tc &expr,
    object_mapt &dest,
    const std::string &suffix,
    const type2tc &original_type) const;

  // Like get_value_set_rec, but dedicated to walking through the ireps that
  // are produced by pointer deref byte stitching
  void get_byte_stitching_value_set(
    const expr2tc &expr,
    object_mapt &dest,
    const std::string &suffix,
    const type2tc &original_type) const;


  /** Internal get_value_set method. Just the same as the other get_value_set
   *  method, but collects into an object_mapt instead of a list of exprs.
   *  @param expr The expression to evaluate the value set of.
   *  @param dest Destination value set object map to store the result into. */
  void get_value_set(
    const expr2tc &expr,
    object_mapt &dest) const;

  /** Internal get_reference_set method. Just the same as the other
   *  get_reference_set method, but collects into an object_mapt instead of a
   *  list of exprs.
   *  @param expr The expression to evaluate the reference set of.
   *  @param dest Destination value set object map to store results into. */
  void get_reference_set(const expr2tc &expr, object_mapt &dest) const
  {
    get_reference_set_rec(expr, dest);
  }

  /** Recursive implementation of get_reference_set.
   *  @param expr The (portion of the) expression we're evaluating the reference
   *         set of.
   *  @param dest Destination value set map to store results into. */
  void get_reference_set_rec(const expr2tc &expr, object_mapt &dest) const;

  /** Recursive assign method implementation -- descends through the left hand
   *  side looking for symbols to assign values to.
   *  @param lhs Left hand side expression that we're assigning value sets
   *         to.
   *  @param values_rhs The value set of the right hand side of the assignment,
   *         i.e. all the things the rhs points at.
   *  @param suffix Accumulated suffix of the lhs up to this point. See docs for
   *         @ref entryt and @get_value_set_rec.
   *  @param add_to_sets See @ref assign. */
  void assign_rec(
    const expr2tc &lhs,
    const object_mapt &values_rhs,
    const std::string &suffix,
    bool add_to_sets);

  /** Mark dynamic objects as (possibly) deallocated, and thus invalid.
   *  Something to do with the black magic that allows the static analysis to
   *  deal with dynamically allocated memory.
   *  @param op Operand evaluating to the pointer to free. */
  void do_free(const expr2tc &op);

  /** Attempt to extract the member of an expression statically. If it's a
   *  chain of with's, or a constant struct, then pick out the actual expression
   *  value of the given struct component. If not, just formulate a member2t
   *  expression.
   *  @param src Structure type'd expr to extract a member from.
   *  @param component_name Name of the component to extract from src. */
  expr2tc make_member(const expr2tc &src, const irep_idt &component_name);

public:
//********************************** Members ***********************************
  /** Some crazy static analysis tool. */
  unsigned location_number;
  /** Object to assign numbers to objects -- i.e., the numbers in the map of
   *  a @ref object_map_dt. Static and bad. */
  static object_numberingt object_numbering;

  /** Storage for all the value sets for all the variables in the program. See
   *  @ref entryt for the format of the string used as an index. */
  valuest values;

  /** Namespace for looking up types against. */
  const namespacet &ns;

  irep_idt xchg_name;
  unsigned long xchg_num;
};

#endif
