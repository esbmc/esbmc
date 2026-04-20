#pragma once

#include <util/namespace.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/symex_target.h>
#include <solvers/smt/smt_conv.h>


class SSA_stept
{
public:
  
 symex_targett::sourcet source;
 goto_trace_stept::typet type;

 // One stack trace recorded per function activation record. Valid for
 // assignment and assert steps only. In reverse order (most recent in idx
 // 0).
 std::vector<stack_framet> stack_trace;

 bool is_assert() const
 {
   return type == goto_trace_stept::ASSERT;
 }
 bool is_assume() const
 {
   return type == goto_trace_stept::ASSUME;
 }
 bool is_assignment() const
 {
   return type == goto_trace_stept::ASSIGNMENT;
 }
 bool is_output() const
 {
   return type == goto_trace_stept::OUTPUT;
 }
 bool is_renumber() const
 {
   return type == goto_trace_stept::RENUMBER;
 }
 bool is_skip() const
 {
   return type == goto_trace_stept::SKIP;
 }
 bool is_branching() const
 {
   return type == goto_trace_stept::BREANCHING;
 }

 expr2tc guard;

 // for ASSIGNMENT
 expr2tc lhs, rhs;
 expr2tc original_lhs, original_rhs;

 // for ASSUME/ASSERT
 expr2tc cond;
 std::string comment;

 // for OUTPUT
 std::string format_string;
 std::list<expr2tc> output_args;

 // for conversion
 smt_astt guard_ast, cond_ast;
 std::list<expr2tc> converted_output_args;

 // for slicing
 bool ignore;

 // for visibility
 bool hidden;

 // for bidirectional search
 unsigned loop_number;

 SSA_stept() : ignore(false), hidden(false)
 {
 }

 void output(const namespacet &ns, std::ostream &out) const;
 void short_output(
		   const namespacet &ns,
		   std::ostream &out,
		   bool show_ignored = false) const;
 void dump() const;
};

typedef std::list<SSA_stept> SSA_stepst;
