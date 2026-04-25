#ifndef CPROVER_GOTO_PROGRAMS_RW_SET
#define CPROVER_GOTO_PROGRAMS_RW_SET

#include <goto-programs/goto_program.h>
#include <util/namespace.h>

class rw_sett
{
public:
  struct entryt
  {
    irep_idt object;
    bool r, w, deref;
    expr2tc guard;
    expr2tc original_expr;

    entryt() : r(false), w(false), deref(false), guard(gen_true_expr())
    {
    }

    const expr2tc &get_guard() const
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

  /// Compute read/write entries for the given instruction-level expression.
  /// Dispatches on the irep2 code kind (assign / printf / return /
  /// function_call) or, for plain expressions on goto/assert/assume
  /// instructions, treats the expression as a read.
  void compute(const expr2tc &expr);

  rw_sett(const namespacet &_ns, goto_programt::const_targett _target)
    : ns(_ns), target(_target)
  {
  }

  rw_sett(
    const namespacet &_ns,
    goto_programt::const_targett _target,
    const expr2tc &expr)
    : ns(_ns), target(_target)
  {
    compute(expr);
  }

  void read_rec(const expr2tc &expr)
  {
    read_write_rec(expr, true, false, "", gen_true_expr(), expr2tc());
  }

  void read_rec(
    const expr2tc &expr,
    const expr2tc &guard,
    const expr2tc &original_expr)
  {
    read_write_rec(expr, true, false, "", guard, original_expr);
  }

protected:
  const namespacet &ns;
  const goto_programt::const_targett target;

  void assign(const expr2tc &lhs, const expr2tc &rhs);

  void read_write_rec(
    const expr2tc &expr,
    bool r,
    bool w,
    const std::string &suffix,
    const expr2tc &guard,
    const expr2tc &original_expr,
    bool dereferenced = false);
};

#define forall_rw_set_entries(it, rw_set)                                      \
  for (rw_sett::entriest::const_iterator it = (rw_set).entries.begin();        \
       it != (rw_set).entries.end();                                           \
       it++)

#endif
