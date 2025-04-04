#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_convert_class.h>
#include <goto-programs/loop_unroll.h>
#include <langapi/language_util.h>
#include <unordered_set>

class goto_coveraget
{
public:
  explicit goto_coveraget(const namespacet &ns, goto_functionst &goto_functions)
    : ns(ns), goto_functions(goto_functions)
  {
    target_num = -1;
  };
  explicit goto_coveraget(
    const namespacet &ns,
    goto_functionst &goto_functions,
    const std::string filename)
    : ns(ns), goto_functions(goto_functions), filename(filename)
  {
    target_num = -1;
  };
  void assertion_coverage();
  void branch_coverage();
  void branch_function_coverage();

  void insert_assert(
    goto_programt &goto_program,
    goto_programt::targett &it,
    const expr2tc &guard);

  // customize comment
  void insert_assert(
    goto_programt &goto_program,
    goto_programt::targett &it,
    const expr2tc &guard,
    const std::string &idf);

  // replace every assertion to a specific guard
  void replace_all_asserts_to_guard(
    const expr2tc &guard,
    bool is_instrumentation = false);
  // replace an assertion to a specific guard
  void replace_assert_to_guard(
    const expr2tc &guard,
    goto_programt::instructiont::targett &it,
    bool is_instrumentation);

  // convert assert(cond) to assert(!cond)
  void negating_asserts(const std::string &tgt_fname);

  // condition cov
  void condition_coverage();
  exprt
  gen_not_eq_expr(const exprt &lhs, const exprt &rhs, const locationt &loc);
  exprt gen_and_expr(const exprt &lhs, const exprt &rhs, const locationt &loc);
  exprt gen_not_expr(const exprt &expr, const locationt &loc);
  expr2tc gen_not_expr(const expr2tc &expr);
  int get_total_instrument() const;
  int get_total_assert_instance() const;
  std::set<std::pair<std::string, std::string>> get_total_cond_assert() const;
  std::string get_filename_from_path(std::string path);
  void set_target(const std::string &_tgt);
  bool is_target_func(const irep_idt &f, const std::string &tgt_name) const;

  // total numbers of instrumentation
  static size_t total_assert;
  static size_t total_assert_ins;
  static std::set<std::pair<std::string, std::string>> total_cond;
  static size_t total_branch;
  static size_t total_func_branch;

  std::string target_function = "";

protected:
  // turn a OP b OP c into a list a, b, c
  exprt handle_single_guard(exprt &expr, bool top_level);
  void handle_operands_guard(
    exprt &expr,
    goto_programt &goto_program,
    goto_programt::instructiont::targett &it);
  void add_cond_cov_assert(
    const exprt &top_ptr,
    const exprt &pre_cond,
    goto_programt &goto_program,
    goto_programt::instructiont::targett &it);
  void gen_cond_cov_assert(
    exprt top_ptr,
    exprt pre_cond,
    goto_programt &goto_program,
    goto_programt::instructiont::targett &it);

  namespacet ns;
  goto_functionst &goto_functions;

  // we need to skip the conditions within the built-in library
  // while keeping the file manually included by user
  // this filter, however, is unsound.. E.g. if the src filename is the same as the builtin library name
  std::string filename;

  int target_num;
};

class goto_coverage_rm : goto_convertt
{
public:
  goto_coverage_rm(
    contextt &_context,
    optionst &_options,
    goto_functionst &goto_functions)
    : goto_convertt(_context, _options), goto_functions(goto_functions)
  {
    options.set_option("goto-instrumented", true);
  }
  void remove_sideeffect();
  goto_functionst &goto_functions;
};
