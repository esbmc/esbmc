#ifndef CPROVER_GOTO_PROGRAMS_RW_SET
#define CPROVER_GOTO_PROGRAMS_RW_SET

#include <pointer-analysis/value_sets.h>
#include <util/expr_util.h>
#include <irep2/irep2_guard.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <unordered_set>

class rw_sett
{
public:
  struct entryt
  {
    irep_idt object;
    bool r, w, deref;
    guard2tc guard;
    expr2tc original_expr;

    entryt() : r(false), w(false), deref(false)
    {
    }

    const guard2tc &get_guard() const
    {
      return guard;
    }

    std::string get_comment() const
    {
      std::string result;
      if (w)
        result = "W/W";
      else
        result = "R/W";

      result += " data race on " + id2string(object);

      return result;
    }
  };

  typedef std::unordered_map<irep_idt, entryt, irep_id_hash> entriest;
  entriest entries;

  // Set of non-static local symbols whose address is taken somewhere in the
  // program. Such locals may alias a pointer handed to another thread, so they
  // remain race-eligible even when accessed directly by name (issue #4424).
  // When null, no local is treated as shared (legacy behaviour).
  typedef std::unordered_set<irep_idt, irep_id_hash> shared_localst;

  void compute(const expr2tc &expr);

  rw_sett(
    const namespacet &_ns,
    goto_programt::const_targett _target,
    const shared_localst *_shared_locals = nullptr)
    : ns(_ns), target(_target), shared_locals(_shared_locals)
  {
  }

  rw_sett(
    const namespacet &_ns,
    goto_programt::const_targett _target,
    const expr2tc &expr,
    const shared_localst *_shared_locals = nullptr)
    : ns(_ns), target(_target), shared_locals(_shared_locals)
  {
    compute(expr);
  }

  void read_rec(const expr2tc &expr)
  {
    read_write_rec(expr, true, false, "", guard2tc(), expr2tc());
  }

  void read_rec(
    const expr2tc &expr,
    const guard2tc &guard,
    const expr2tc &original_expr)
  {
    read_write_rec(expr, true, false, "", guard, original_expr);
  }

protected:
  const namespacet &ns;
  const goto_programt::const_targett target;
  const shared_localst *shared_locals;

  void assign(const expr2tc &lhs, const expr2tc &rhs);

  void read_write_rec(
    const expr2tc &expr,
    bool r,
    bool w,
    const std::string &suffix,
    const guard2tc &guard,
    const expr2tc &original_expr,
    bool dereferenced = false);
};

#define forall_rw_set_entries(it, rw_set)                                      \
  for (rw_sett::entriest::const_iterator it = (rw_set).entries.begin();        \
       it != (rw_set).entries.end();                                           \
       it++)

#endif
