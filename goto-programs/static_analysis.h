/*******************************************************************\

Module: Static Analysis

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_STATIC_ANALYSIS_H
#define CPROVER_GOTO_PROGRAMS_STATIC_ANALYSIS_H

#include <irep2.h>
#include <map>
#include <iostream>

#include "goto_functions.h"

// don't use me -- I am just a base class
// please derive from me
class abstract_domain_baset
{
public:
  abstract_domain_baset():seen(false)
  {
  }

  typedef goto_programt::const_targett locationt;

  virtual void initialize(
    const namespacet &ns,
    locationt l)=0;

  // how function calls are treated:
  // a) there is an edge from each call site to the function head
  // b) there is an edge from each return to the last instruction (END_FUNCTION)
  //    of each function
  // c) there is an edge from the last instruction of the function
  //    to the instruction following the call site
  //    (for setting the LHS)

  virtual void transform(
    const namespacet &ns,
    locationt from,
    locationt to)=0;

  virtual ~abstract_domain_baset()
  {
  }
  
  virtual void output(
    const namespacet &ns __attribute__((unused)),
    std::ostream &out __attribute__((unused))) const
  {
  }
  
  typedef hash_set_cont<exprt, irep_hash> expr_sett;
  
  virtual void get_reference_set(
    const namespacet &ns __attribute__((unused)),
    const expr2tc &expr __attribute__((unused)),
    std::list<expr2tc> &dest __attribute__((unused))) { assert(0); };
  
  // also add
  //
  //   bool merge(const T &b);
  //
  // returns true iff there is s.th. new
  
protected:
  bool seen;
  
  friend class static_analysis_baset;

  // utilities  
  
  // get guard of a conditional edge
  expr2tc get_guard(locationt from, locationt to) const;
  
  // get lhs that return value is assigned to
  // for an edge that returns from a function
  expr2tc get_return_lhs(locationt to) const;
};

// don't use me -- I am just a base class
// use static_analysist instead
class static_analysis_baset
{
public:
  typedef abstract_domain_baset statet;
  typedef goto_programt::const_targett locationt;

  static_analysis_baset(const namespacet &_ns):
    ns(_ns),
    initialized(false)
  {
  }
  
  virtual void initialize(
    const goto_programt &goto_program)
  {
    if(!initialized)
    {
      initialized=true;
      generate_states(goto_program);
    }
  }
    
  virtual void initialize(
    const goto_functionst &goto_functions)
  {
    if(!initialized)
    {
      initialized=true;
      generate_states(goto_functions);
    }
  }
    
  virtual void update(const goto_programt &goto_program);
  virtual void update(const goto_functionst &goto_functions);
    
  virtual void operator()(
    const goto_programt &goto_program);
    
  virtual void operator()(
    const goto_functionst &goto_functions);

  virtual ~static_analysis_baset()
  {
  }

  virtual void clear()
  {
    initialized=false;
  }
  
  virtual void output(
    const goto_functionst &goto_functions,
    std::ostream &out) const;

  void output(
    const goto_programt &goto_program,
    std::ostream &out) const
  {
    output(goto_program, "", out);
  }

  virtual bool has_location(locationt l) const=0;
  
  void insert(locationt l)
  {
    generate_state(l);
  }

protected:
  const namespacet &ns;
  
  virtual void output(
    const goto_programt &goto_program,
    const irep_idt &identifier,
    std::ostream &out) const;

  typedef std::map<unsigned, locationt> working_sett;
  
  locationt get_next(working_sett &working_set);
  
  void put_in_working_set(
    working_sett &working_set,
    locationt l)
  {
    working_set.insert(
      std::pair<unsigned, locationt>(l->location_number, l));
  }

  // true = found s.th. new
  bool fixedpoint(
    const goto_programt &goto_program,
    const goto_functionst &goto_functions);
    
  bool fixedpoint(
    goto_functionst::function_mapt::const_iterator it,
    const goto_functionst &goto_functions);
    
  void fixedpoint(
    const goto_functionst &goto_functions);

  // true = found s.th. new
  bool visit(
    locationt l,
    working_sett &working_set,
    const goto_programt &goto_program,
    const goto_functionst &goto_functions);
    
  static locationt successor(locationt l)
  {
    l++;
    return l;
  }
  
  virtual bool merge(statet &a, const statet &b, bool keepnew=false)=0;
  
  typedef std::set<irep_idt> functions_donet;
  functions_donet functions_done;

  typedef std::set<irep_idt> recursion_sett;
  recursion_sett recursion_set;
  
  void generate_states(
    const goto_functionst &goto_functions);

  void generate_states(
    const goto_programt &goto_program);
    
  bool initialized;
  
  // function calls
  void do_function_call_rec(
    locationt l_call,
    const expr2tc &function,
    const std::vector<expr2tc> &arguments,
    statet &new_state,
    const goto_functionst &goto_functions);

  void do_function_call(
    locationt l_call,
    const goto_functionst &goto_functions,
    const goto_functionst::function_mapt::const_iterator f_it,
    const std::vector<expr2tc> &arguments,
    statet &new_state);

  // abstract methods
    
  virtual void generate_state(locationt l)=0;
  virtual statet &get_state(locationt l)=0;
  virtual const statet &get_state(locationt l) const=0;
  virtual statet* make_temporary_state(statet &s)=0;

  typedef abstract_domain_baset::expr_sett expr_sett;

  virtual void get_reference_set(
    locationt l,
    const expr2tc &expr,
    std::list<expr2tc> &dest)=0;
};

// T is expected to be derived from abstract_domain_baset
template<typename T>
class static_analysist:public static_analysis_baset
{
public:
  // constructor
  static_analysist(const namespacet &_ns):
    static_analysis_baset(_ns)
  {
  }

  typedef goto_programt::const_targett locationt;

  inline T &operator[](locationt l)
  {
    typename state_mapt::iterator it=state_map.find(l);
    if(it==state_map.end()) throw "failed to find state";
    return it->second;
  }
    
  inline const T &operator[](locationt l) const
  {
    typename state_mapt::const_iterator it=state_map.find(l);
    if(it==state_map.end()) throw "failed to find state";
    return it->second;
  }
  
  virtual void clear()
  {
    state_map.clear();
    static_analysis_baset::clear();
  }

  virtual bool has_location(locationt l) const
  {
    return state_map.count(l)!=0;
  }
  
protected:
  typedef std::map<locationt, T> state_mapt;
  state_mapt state_map;

  virtual statet &get_state(locationt l)
  {
    typename state_mapt::iterator it=state_map.find(l);
    if(it==state_map.end()) throw "failed to find state";
    return it->second;
  }

  virtual const statet &get_state(locationt l) const
  {
    typename state_mapt::const_iterator it=state_map.find(l);
    if(it==state_map.end()) throw "failed to find state";
    return it->second;
  }

  virtual bool merge(statet &a, const statet &b, bool keepnew=false)
  {
    return static_cast<T &>(a).merge(static_cast<const T &>(b), keepnew);
  }
  
  virtual statet* make_temporary_state(statet &s)
  {
    return new T(static_cast<T &>(s));
  }

  virtual void generate_state(locationt l)
  {
    state_map[l].initialize(ns, l);
  }

  virtual void get_reference_set(
    locationt l,
    const expr2tc &expr,
    std::list<expr2tc> &dest)
  {
    state_map[l].get_reference_set(ns, expr, dest);
  }
};

#endif
