#include <goto-programs/goto_functions.h>
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
  // add an assert(0)
  // - at the beginning of each GOTO program
  // - at the beginning of each branch body
  // - before each END_FUNCTION statement
  void add_false_asserts();

  void insert_assert(
    goto_programt &goto_program,
    goto_programt::targett &it,
    const expr2tc &guard);

  // convert every assertion to an assert(0)
  void make_asserts_false();

  // convert every assertion to an assert(1)
  void make_asserts_true();

  // condition cov
  void gen_cond_cov();
  int get_total_instrument() const;
  int get_total_assert_instance() const;
  std::unordered_set<std::string> get_total_cond_assert() const;
  std::string get_filename_from_path(std::string path);

protected:
  // turn a OP b OP c into a list a, b, c
  static void
  collect_operands(const exprt &expr, std::list<exprt> &operands, bool &flag);
  static void
  collect_operators(const exprt &expr, std::list<std::string> &operators);
  static void collect_atom_operands(const exprt &expr, std::set<exprt> &atoms);
  exprt handle_single_guard(exprt &guard, bool &flag);
  void add_cond_cov_init_assert(
    const exprt &expr,
    goto_programt &goto_program,
    goto_programt::targett &it);
  void add_cond_cov_rhs_assert(
    const irep_idt &op_tp,
    exprt::operandst::iterator &top_ptr,
    exprt::operandst::iterator &rhs_ptr,
    const exprt::operandst::iterator &root_ptr,
    const exprt &rhs,
    goto_programt &goto_program,
    goto_programt::targett &it);

  namespacet ns;
  goto_functionst &goto_functions;
  std::string filename;
  int target_num;
};
