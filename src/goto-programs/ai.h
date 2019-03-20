/*******************************************************************\

Module: Abstract Interpretation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

/// \file
/// Abstract Interpretation

#ifndef CPROVER_ANALYSES_AI_H
#define CPROVER_ANALYSES_AI_H

#include <iosfwd>
#include <map>
#include <memory>
#include <goto-programs/ai_domain.h>
#include <goto-programs/goto_functions.h>
#include <util/xml.h>
#include <util/expr.h>

// This is a stand-in for std::make_unique, which isn't part of the standard
// library until C++14.  When we move to C++14, we should do a find-and-replace
// on this to use std::make_unique instead.

template <typename T, typename... Ts>
static inline std::unique_ptr<T> util_make_unique(Ts &&... ts)
{
  return std::unique_ptr<T>(new T(std::forward<Ts>(ts)...));
}

/// The basic interface of an abstract interpreter.  This should be enough
/// to create, run and query an abstract interpreter.
// don't use me -- I am just a base class
// use ait instead
class ai_baset
{
public:
  typedef ai_domain_baset statet;

  ai_baset()
  {
  }

  virtual ~ai_baset()
  {
  }

  /// Running the interpreter
  void operator()(const goto_programt &goto_program, const namespacet &ns)
  {
    goto_functionst goto_functions;
    initialize(goto_program);
    entry_state(goto_program);
    fixedpoint(goto_program, goto_functions, ns);
    finalize();
  }

  void operator()(const goto_functionst &goto_functions, const namespacet &ns)
  {
    initialize(goto_functions);
    entry_state(goto_functions);
    fixedpoint(goto_functions, ns);
    finalize();
  }

  /// Accessing individual domains at particular locations
  /// (without needing to know what kind of domain or history is used)
  /// A pointer to a copy as the method should be const and
  /// there are some non-trivial cases including merging domains, etc.
  /// Intended for users of the abstract interpreter; don't use internally.

  /// Returns the abstract state before the given instruction
  /// PRECONDITION(l is dereferenceable)
  virtual std::unique_ptr<statet>
  abstract_state_before(goto_programt::const_targett l) const = 0;

  /// Returns the abstract state after the given instruction
  virtual std::unique_ptr<statet>
  abstract_state_after(goto_programt::const_targett l) const
  {
    /// PRECONDITION(l is dereferenceable && std::next(l) is dereferenceable)
    /// Check relies on a DATA_INVARIANT of goto_programs
    assert(!l->is_end_function()); // No state after the last instruction
    return abstract_state_before(std::next(l));
  }

  /// Resets the domain
  virtual void clear()
  {
  }

  virtual void
  output(const goto_functionst &goto_functions, std::ostream &out) const;

protected:
  // overload to add a factory
  virtual void initialize(const goto_programt &);
  virtual void initialize(const goto_functiont &);
  virtual void initialize(const goto_functionst &);

  // override to add a cleanup step after fixedpoint has run
  virtual void finalize();

  void entry_state(const goto_programt &);
  void entry_state(const goto_functionst &);

  // the work-queue is sorted by location number
  typedef hash_map_cont<unsigned, goto_programt::const_targett> working_sett;

  goto_programt::const_targett get_next(working_sett &working_set);

  void
  put_in_working_set(working_sett &working_set, goto_programt::const_targett l)
  {
    working_set.insert(
      std::pair<unsigned, goto_programt::const_targett>(l->location_number, l));
  }

  // true = found something new
  bool fixedpoint(
    const goto_programt &goto_program,
    const goto_functionst &goto_functions,
    const namespacet &ns);

  virtual void
  fixedpoint(const goto_functionst &goto_functions, const namespacet &ns) = 0;

  void sequential_fixedpoint(
    const goto_functionst &goto_functions,
    const namespacet &ns);

  // Visit performs one step of abstract interpretation from location l
  // Depending on the instruction type it may compute a number of "edges"
  // or applications of the abstract transformer
  // true = found something new
  bool visit(
    goto_programt::const_targett l,
    working_sett &working_set,
    const goto_programt &goto_program,
    const goto_functionst &goto_functions,
    const namespacet &ns);

  // function calls
  bool do_function_call_rec(
    goto_programt::const_targett l_call,
    goto_programt::const_targett l_return,
    const expr2tc &function,
    const goto_functionst &goto_functions,
    const namespacet &ns);

  bool do_function_call(
    goto_programt::const_targett l_call,
    goto_programt::const_targett l_return,
    const goto_functionst &goto_functions,
    const goto_functionst::function_mapt::const_iterator f_it,
    const namespacet &ns);

  // abstract methods

  virtual bool merge(
    const statet &src,
    goto_programt::const_targett from,
    goto_programt::const_targett to) = 0;
  // for concurrent fixedpoint
  virtual bool merge_shared(
    const statet &src,
    goto_programt::const_targett from,
    goto_programt::const_targett to,
    const namespacet &ns) = 0;
  virtual statet &get_state(goto_programt::const_targett l) = 0;
  virtual const statet &find_state(goto_programt::const_targett l) const = 0;
  virtual std::unique_ptr<statet> make_temporary_state(const statet &s) = 0;
};

// domainT is expected to be derived from ai_domain_baseT
template <typename domainT>
class ait : public ai_baset
{
public:
  // constructor
  ait() : ai_baset()
  {
  }

  domainT &operator[](goto_programt::const_targett l)
  {
    typename state_mapt::iterator it = state_map.find(l);
    if(it == state_map.end())
      throw "failed to find state";

    return it->second;
  }

  const domainT &operator[](goto_programt::const_targett l) const
  {
    typename state_mapt::const_iterator it = state_map.find(l);
    if(it == state_map.end())
      throw "failed to find state";

    return it->second;
  }

  std::unique_ptr<statet>
  abstract_state_before(goto_programt::const_targett t) const override
  {
    typename state_mapt::const_iterator it = state_map.find(t);
    if(it == state_map.end())
    {
      std::unique_ptr<statet> d = util_make_unique<domainT>();
      assert(d->is_bottom());
      return d;
    }

    return util_make_unique<domainT>(it->second);
  }

  void clear() override
  {
    state_map.clear();
    ai_baset::clear();
  }

protected:
  typedef hash_map_cont<
    goto_programt::const_targett,
    domainT,
    const_target_hash,
    pointee_address_equalt>
    state_mapt;
  state_mapt state_map;

  // this one creates states, if need be
  virtual statet &get_state(goto_programt::const_targett l) override
  {
    return state_map[l]; // calls default constructor
  }

  // this one just finds states
  const statet &find_state(goto_programt::const_targett l) const override
  {
    typename state_mapt::const_iterator it = state_map.find(l);
    if(it == state_map.end())
      throw "failed to find state";

    return it->second;
  }

  bool merge(
    const statet &src,
    goto_programt::const_targett from,
    goto_programt::const_targett to) override
  {
    statet &dest = get_state(to);
    return static_cast<domainT &>(dest).merge(
      static_cast<const domainT &>(src), from, to);
  }

  std::unique_ptr<statet> make_temporary_state(const statet &s) override
  {
    return util_make_unique<domainT>(static_cast<const domainT &>(s));
  }

  void fixedpoint(const goto_functionst &goto_functions, const namespacet &ns)
    override
  {
    sequential_fixedpoint(goto_functions, ns);
  }

private:
  // to enforce that domainT is derived from ai_domain_baset
  void dummy(const domainT &s)
  {
    const statet &x = s;
    (void)x;
  }

  // not implemented in sequential analyses
  bool merge_shared(
    const statet &,
    goto_programt::const_targett,
    goto_programt::const_targett,
    const namespacet &) override
  {
    throw "not implemented";
  }
};

#endif // CPROVER_ANALYSES_AI_H
