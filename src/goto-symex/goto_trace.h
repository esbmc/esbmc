#ifndef CPROVER_GOTO_SYMEX_GOTO_TRACE_H
#define CPROVER_GOTO_SYMEX_GOTO_TRACE_H

#include <fstream>
#include <goto-programs/goto_program.h>
#include <goto-symex/symex_target.h>

#include <map>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/cmdline.h>
#include <vector>
#include <string_view>

struct function_coveraget {
  std::string name;
  std::string file;
  int start_line;
  int end_line;
  std::set<int> covered_lines;
  std::map<int, int> line_hits;

  function_coveraget() : start_line(0), end_line(0) {}
};

struct file_coveraget {
  std::string name;
  std::map<std::string, function_coveraget> functions;
  std::set<int> covered_lines;
};

// Forward declare goto_tracet
class goto_tracet;

class goto_trace_stept
{
public:
  unsigned step_nr;

  // See SSA_stept.
  std::vector<stack_framet> stack_trace;

  bool is_assignment() const
  {
    return type == ASSIGNMENT;
  }
  bool is_assume() const
  {
    return type == ASSUME;
  }
  bool is_assert() const
  {
    return type == ASSERT;
  }
  bool is_output() const
  {
    return type == OUTPUT;
  }
  bool is_skip() const
  {
    return type == SKIP;
  }
  bool is_renumber() const
  {
    return type == RENUMBER;
  }

  typedef enum
  {
    ASSIGNMENT,
    ASSUME,
    ASSERT,
    OUTPUT,
    SKIP,
    RENUMBER
  } typet;
  typet type;

  goto_programt::const_targett pc;

  // this transition done by given thread number
  unsigned thread_nr;

  // for assume, assert, goto
  bool guard;

  // for assert
  std::string comment;

  // in SSA
  expr2tc lhs, rhs;

  // this is a constant
  expr2tc value;

  // original expression
  expr2tc original_lhs;

  // for OUTPUT
  std::string format_string;
  std::list<expr2tc> output_args;

  void track_coverage(goto_tracet const& trace) const;

  void output(const class namespacet &ns, std::ostream &out) const;
  void dump() const;

  goto_trace_stept() : step_nr(0), thread_nr(0), guard(false)
  {
  }
};

class goto_tracet
{
public:
  typedef std::list<goto_trace_stept> stepst;
  typedef std::map<std::string, std::string, std::less<std::string>> mid;
  // stepst steps;
  std::string mode;
  std::vector<goto_trace_stept> steps;
  mutable std::map<std::string, file_coveraget> coverage_data;

  void update_coverage(const goto_trace_stept& step) const;

  void clear()
  {
    mode.clear();
    steps.clear();
  }

  void output(const class namespacet &ns, std::ostream &out) const;
};

void show_goto_trace_gui(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace);

void show_goto_trace(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace);

void violation_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace);

void correctness_graphml_goto_trace(
  optionst &options,
  const namespacet &ns,
  const goto_tracet &goto_trace);

void generate_goto_trace_in_correctness_graphml_format(
  std::string &witness_output,
  bool is_detailed_mode,
  int &specification,
  const namespacet &ns,
  const goto_tracet &goto_trace);

void counterexample_value(
  std::ostream &out,
  const namespacet &ns,
  const expr2tc &identifier,
  const expr2tc &value);

void generate_html_report(
  const std::string_view uuid,
  const namespacet &ns,
  const goto_tracet &goto_trace,
  const cmdlinet::options_mapt &options_map);

#endif
