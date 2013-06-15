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

class value_sett
{
public:
  value_sett():location_number(0)
  {
  }

//*********************************** Types ************************************

  typedef std::set<expr2tc> expr_sett;

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

    mp_integer offset;
    bool offset_is_set;
    bool offset_is_zero() const
    { return offset_is_set && offset.is_zero(); }
  };

  typedef std::map<unsigned, objectt> obj_map_mapt;
  class object_map_dt
  {
    // If you said this class looks pretty map like, it's because it used to be
    // a subclass of std::map, which was far too funky for me, thanks.
  public:
    ~object_map_dt()
    {
      for (obj_map_mapt::const_iterator it = themap.begin();
           it != themap.end(); it++) {
        value_sett::obj_numbering_deref(it->first);
      }
    }

    object_map_dt()
    {
    }

    object_map_dt(const object_map_dt &ref)
    {
      *this = ref;
      for (obj_map_mapt::const_iterator it = themap.begin();
           it != themap.end(); it++) {
        value_sett::obj_numbering_ref(it->first);
      }
    }

    typedef obj_map_mapt::const_iterator const_iterator;
    typedef obj_map_mapt::iterator iterator;

    objectt &operator[](unsigned i)
    {
      if (themap.find(i) == themap.end())
        value_sett::obj_numbering_ref(i);
      return themap[i];
    }

    const_iterator find(unsigned i) const
    {
      return themap.find(i);
    }

    iterator find(unsigned i)
    {
      return themap.find(i);
    }

    const_iterator begin() const
    {
      return themap.begin();
    }

    const_iterator end() const
    {
      return themap.end();
    }

    size_t size(void) const
    {
      return themap.size();
    }

    const static object_map_dt empty;
    obj_map_mapt themap;
  };

  typedef reference_counting<object_map_dt> object_mapt;

  struct entryt
  {
    object_mapt object_map;
    std::string identifier;
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

  typedef hash_map_cont<string_wrapper, entryt, string_wrap_hash> valuest;

//********************************** Methods ***********************************

  expr2tc to_expr(object_map_dt::const_iterator it) const;

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

  bool erase(const std::string &name)
  {
    return (values.erase(string_wrapper(name)) == 1);
  }

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

  static void obj_numbering_ref(unsigned int num);
  static void obj_numbering_deref(unsigned int num);

public:
//********************************** Members ***********************************
  unsigned location_number;
  static object_numberingt object_numbering;
  static object_number_numberingt obj_numbering_refset;

  valuest values;
};

#endif
