#ifndef CPROVER_GOTO_PROGRAMS_GOTO_CONVERT_CLASS_H
#define CPROVER_GOTO_PROGRAMS_GOTO_CONVERT_CLASS_H

#include <goto-programs/goto_program.h>
#include <list>
#include <queue>
#include <stack>
#include <util/expr_util.h>
#include <util/guard.h>
#include <util/namespace.h>
#include <util/options.h>
#include <util/symbol_generator.h>
#include <util/std_code.h>

class goto_convertt
{
public:
  void goto_convert(const codet &code, goto_programt &dest);

  goto_convertt(contextt &_context, optionst &_options)
    : context(_context),
      options(_options),
      ns(_context),
      tmp_symbol("goto_convertt::")
  {
  }

protected:
  contextt &context;
  optionst &options;
  namespacet ns;
  symbol_generator tmp_symbol;

  void goto_convert_rec(const codet &code, goto_programt &dest);

  //
  // tools for symbols
  //
  void new_name(symbolt &symbol);

  symbolt &new_tmp_symbol(const typet &type);
  symbolt &new_cftest_symbol(const typet &type);

  //
  // side effect removal
  //
  void make_temp_symbol(exprt &expr, goto_programt &dest);
  unsigned int get_expr_number_globals(const exprt &expr);
  unsigned int get_expr_number_globals(const expr2tc &expr);
  void break_globals2assignments(
    exprt &rhs,
    goto_programt &dest,
    const locationt &location);
  void break_globals2assignments(
    int &atomic,
    exprt &lhs,
    exprt &rhs,
    goto_programt &dest,
    const locationt &location);
  void break_globals2assignments(
    int &atomic,
    exprt &rhs,
    goto_programt &dest,
    const locationt &location);
  void break_globals2assignments_rec(
    exprt &rhs,
    exprt &atomic_dest,
    goto_programt &dest,
    int atomic,
    const locationt &location);

  // this produces if(guard) dest;
  void guard_program(const guardt &guard, goto_programt &dest);

  void remove_sideeffects(
    exprt &expr,
    goto_programt &dest,
    bool result_is_used = true);

  void address_of_replace_objects(exprt &expr, goto_programt &dest);

  bool rewrite_vla_decl(typet &var_type, goto_programt &dest);
  bool rewrite_vla_decl_size(exprt &size, goto_programt &dest);
  void generate_dynamic_size_vla(
    exprt &var,
    const locationt &loc,
    goto_programt &dest);

  bool has_sideeffect(const exprt &expr);

  void remove_assignment(exprt &expr, goto_programt &dest, bool result_is_used);
  void remove_post(exprt &expr, goto_programt &dest, bool result_is_used);
  void remove_pre(exprt &expr, goto_programt &dest, bool result_is_used);
  void
  remove_function_call(exprt &expr, goto_programt &dest, bool result_is_used);
  void remove_cpp_new(exprt &expr, goto_programt &dest, bool result_is_used);
  void remove_cpp_delete(exprt &expr, goto_programt &dest);
  void remove_temporary_object(exprt &expr, goto_programt &dest);
  void remove_statement_expression(
    exprt &expr,
    goto_programt &dest,
    bool result_is_used);
  void remove_gcc_conditional_expression(exprt &expr, goto_programt &dest);

  virtual void
  do_cpp_new(const exprt &lhs, const exprt &rhs, goto_programt &dest);

  static void replace_new_object(const exprt &object, exprt &dest);

  void
  cpp_new_initializer(const exprt &lhs, const exprt &rhs, goto_programt &dest);

  //
  // function calls
  //

  virtual void do_function_call(
    const exprt &lhs,
    const exprt &function,
    const exprt::operandst &arguments,
    goto_programt &dest);

  virtual void do_function_call_if(
    const exprt &lhs,
    const exprt &function,
    const exprt::operandst &arguments,
    goto_programt &dest);

  virtual void do_function_call_symbol(
    const exprt &lhs,
    const exprt &function,
    const exprt::operandst &arguments,
    goto_programt &dest);

  virtual void do_function_call_dereference(
    const exprt &lhs,
    const exprt &function,
    const exprt::operandst &arguments,
    goto_programt &dest);

  //
  // conversion
  //
  void convert_block(const codet &code, goto_programt &dest);
  void convert_decl(const codet &code, goto_programt &dest);
  void convert_decl_block(const codet &code, goto_programt &dest);
  void convert_expression(const codet &code, goto_programt &dest);
  void convert_assign(const code_assignt &code, goto_programt &dest);
  void convert_cpp_delete(const codet &code, goto_programt &dest);
  void convert_for(const codet &code, goto_programt &dest);
  void convert_while(const codet &code, goto_programt &dest);
  void convert_dowhile(const codet &code, goto_programt &dest);
  void convert_assume(const codet &code, goto_programt &dest);
  void convert_assert(const codet &code, goto_programt &dest);
  void convert_switch(const codet &code, goto_programt &dest);
  void convert_break(const code_breakt &code, goto_programt &dest);
  void convert_return(const code_returnt &code, goto_programt &dest);
  void convert_continue(const code_continuet &code, goto_programt &dest);
  void convert_ifthenelse(const codet &code, goto_programt &dest);
  void convert_init(const codet &code, goto_programt &dest);
  void convert_goto(const codet &code, goto_programt &dest);
  void convert_skip(const codet &code, goto_programt &dest);
  void convert_non_deterministic_goto(const codet &code, goto_programt &dest);
  void convert_label(const code_labelt &code, goto_programt &dest);
  void convert_switch_case(const code_switch_caset &code, goto_programt &dest);
  void
  convert_function_call(const code_function_callt &code, goto_programt &dest);
  void convert_atomic_begin(const codet &code, goto_programt &dest);
  void convert_atomic_end(const codet &code, goto_programt &dest);
  void convert(const codet &code, goto_programt &dest);
  void copy(
    const codet &code,
    goto_program_instruction_typet type,
    goto_programt &dest);

  typedef std::vector<codet> destructor_stackt;

  void unwind_destructor_stack(
    const locationt &,
    std::size_t stack_size,
    goto_programt &dest);
  void unwind_destructor_stack(
    const locationt &,
    std::size_t stack_size,
    goto_programt &dest,
    destructor_stackt &stack);

  //
  // Try-catch conversion
  //

  void convert_catch(const codet &code, goto_programt &dest);
  void convert_throw(const exprt &expr, goto_programt &dest);
  void convert_throw_decl(const exprt &expr, goto_programt &dest);
  void convert_throw_decl_end(const exprt &expr, goto_programt &dest);

  //
  // gotos
  //

  void finish_gotos(goto_programt &dest);
  void optimize_guarded_gotos(goto_programt &dest);

  typedef std::
    map<irep_idt, std::pair<goto_programt::targett, destructor_stackt>>
      labelst;
  typedef std::list<std::pair<goto_programt::targett, destructor_stackt>>
    gotost;
  typedef std::list<goto_programt::targett> computed_gotost;
  typedef exprt::operandst caset;
  typedef std::list<std::pair<goto_programt::targett, caset>> casest;
  typedef std::map<goto_programt::targett, casest::iterator> cases_mapt;

  struct targetst
  {
    bool return_set, has_return_value, break_set, continue_set, default_set,
      throw_set, leave_set;

    labelst labels;
    gotost gotos;
    computed_gotost computed_gotos;
    destructor_stackt destructor_stack;

    casest cases;
    cases_mapt cases_map;

    goto_programt::targett return_target, break_target, continue_target,
      default_target, throw_target, leave_target;

    std::size_t break_stack_size, continue_stack_size, throw_stack_size,
      leave_stack_size;

    targetst()
      : return_set(false),
        has_return_value(false),
        break_set(false),
        continue_set(false),
        default_set(false),
        throw_set(false),
        leave_set(false),
        break_stack_size(0),
        continue_stack_size(0),
        throw_stack_size(0),
        leave_stack_size(0)
    {
    }

    void set_break(goto_programt::targett _break_target)
    {
      break_set = true;
      break_target = _break_target;
      break_stack_size = destructor_stack.size();
    }

    void set_continue(goto_programt::targett _continue_target)
    {
      continue_set = true;
      continue_target = _continue_target;
      continue_stack_size = destructor_stack.size();
    }

    void set_default(goto_programt::targett _default_target)
    {
      default_set = true;
      default_target = _default_target;
    }

    void set_return(goto_programt::targett _return_target)
    {
      return_set = true;
      return_target = _return_target;
    }

    void set_throw(goto_programt::targett _throw_target)
    {
      throw_set = true;
      throw_target = _throw_target;
      throw_stack_size = destructor_stack.size();
    }

    void set_leave(goto_programt::targett _leave_target)
    {
      leave_set = true;
      leave_target = _leave_target;
      leave_stack_size = destructor_stack.size();
    }
  } targets;

  struct break_continue_targetst
  {
    // for 'while', 'for', 'dowhile'

    explicit break_continue_targetst(const targetst &targets)
    {
      break_set = targets.break_set;
      continue_set = targets.continue_set;
      break_target = targets.break_target;
      continue_target = targets.continue_target;
    }

    void restore(targetst &targets)
    {
      targets.break_set = break_set;
      targets.continue_set = continue_set;
      targets.break_target = break_target;
      targets.continue_target = continue_target;
    }

    goto_programt::targett break_target;
    goto_programt::targett continue_target;
    bool break_set, continue_set;
  };

  struct break_switch_targetst
  {
    // for 'switch'

    explicit break_switch_targetst(const targetst &targets)
    {
      break_set = targets.break_set;
      default_set = targets.default_set;
      break_target = targets.break_target;
      default_target = targets.default_target;
      break_stack_size = targets.destructor_stack.size();
      cases = targets.cases;
      cases_map = targets.cases_map;
    }

    void restore(targetst &targets)
    {
      targets.break_set = break_set;
      targets.default_set = default_set;
      targets.break_target = break_target;
      targets.default_target = default_target;
      targets.cases = cases;
      targets.cases_map = cases_map;
    }

    goto_programt::targett break_target;
    goto_programt::targett default_target;
    bool break_set, default_set;
    std::size_t break_stack_size;

    casest cases;
    cases_mapt cases_map;
  };

  struct throw_targett
  {
    // for 'try...catch' and the like

    explicit throw_targett(const targetst &targets)
    {
      throw_set = targets.throw_set;
      throw_target = targets.throw_target;
      throw_stack_size = targets.destructor_stack.size();
    }

    void restore(targetst &targets)
    {
      targets.throw_set = throw_set;
      targets.throw_target = throw_target;
    }

    goto_programt::targett throw_target;
    bool throw_set;
    std::size_t throw_stack_size;
  };

  void case_guard(const exprt &value, const caset &case_op, exprt &dest);

  // if(cond) { true_case } else { false_case }
  void generate_ifthenelse(
    const exprt &cond,
    goto_programt &true_case,
    goto_programt &false_case,
    const locationt &location,
    goto_programt &dest);

  // if(guard) goto target_true; else goto target_false;
  void generate_conditional_branch(
    const exprt &guard,
    goto_programt::targett target_true,
    goto_programt::targett target_false,
    const locationt &location,
    goto_programt &dest);

  // if(guard) goto target;
  void generate_conditional_branch(
    const exprt &guard,
    goto_programt::targett target_true,
    const locationt &location,
    goto_programt &dest);

  // turn a OP b OP c into a list a, b, c
  static void collect_operands(
    const exprt &expr,
    const irep_idt &id,
    std::list<exprt> &dest);

  // some built-in functions
  void do_abort(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_atomic_begin(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_atomic_end(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_create_thread(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_malloc(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_realloc(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_alloca(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_free(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_sync(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_exit(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_printf(
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
  void do_mem(
    bool is_malloc,
    const exprt &lhs,
    const exprt &rhs,
    const exprt::operandst &arguments,
    goto_programt &dest);
};

#endif
