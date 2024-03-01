#include <goto-programs/goto_functions.h>
#include <goto-programs/loop_unroll.h>
#include <langapi/language_util.h>
#include <unordered_set>

class goto_coveraget
{
public:
  explicit goto_coveraget(namespacet &ns) : ns(ns){};
  // add an assert(0)
  // - at the beginning of each GOTO program
  // - at the beginning of each branch body
  // - before each END_FUNCTION statement
  void add_false_asserts(goto_functionst &goto_functions);

  void insert_assert(
    goto_programt &goto_program,
    goto_programt::targett &it,
    const expr2tc &guard);

  // convert every assertion to an assert(0)
  void make_asserts_false(goto_functionst &goto_functions);

  // convert every assertion to an assert(1)
  void make_asserts_true(goto_functionst &goto_functions);

  // condition cov
  void add_cond_cov_assert(goto_functionst &goto_functions);

  int get_total_instrument() const;

  void count_assert_instance(goto_functionst goto_functions);
  int get_total_assert_instance() const;
  std::unordered_set<std::string> get_total_cond_assert() const;

protected:
  static int total_instrument;
  static int total_assert_instance;
  static std::unordered_set<std::string> total_cond_assert;
  namespacet ns;
};
