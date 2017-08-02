/*******************************************************************\

Module: Static Analysis

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_STATIC_ANALYSIS_H
#define CPROVER_GOTO_PROGRAMS_STATIC_ANALYSIS_H

#include <goto-programs/goto_functions.h>
#include <iostream>
#include <map>
#include <util/irep2.h>

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

  virtual ~abstract_domain_baset() = default;
  
  virtual void output(
    const namespacet &ns __attribute__((unused)),
    std::ostream &out __attribute__((unused))) const
  {
  }

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

  virtual ~static_analysis_baset() = default;

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
  
  void clear() override 
  {
    state_map.clear();
    static_analysis_baset::clear();
  }

  bool has_location(locationt l) const override 
  {
    return state_map.count(l)!=0;
  }
  
protected:
  typedef std::map<locationt, T> state_mapt;
  state_mapt state_map;

  statet &get_state(locationt l) override 
  {
    typename state_mapt::iterator it=state_map.find(l);
    if(it==state_map.end()) throw "failed to find state";
    return it->second;
  }

  const statet &get_state(locationt l) const override 
  {
    typename state_mapt::const_iterator it=state_map.find(l);
    if(it==state_map.end()) throw "failed to find state";
    return it->second;
  }

  bool merge(statet &a, const statet &b, bool keepnew=false) override 
  {
    return static_cast<T &>(a).merge(static_cast<const T &>(b), keepnew);
  }
  
  statet* make_temporary_state(statet &s) override 
  {
    return new T(static_cast<T &>(s));
  }

  void generate_state(locationt l) override 
  {
    state_map[l].initialize(ns, l);
  }

  void get_reference_set(
    locationt l,
    const expr2tc &expr,
    std::list<expr2tc> &dest) override 
  {
    state_map[l].get_reference_set(ns, expr, dest);
  }
};

#endif
