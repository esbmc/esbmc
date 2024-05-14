#include <goto-programs/goto_functions.h>
#include <goto-programs/loop_unroll.h>
#include <langapi/language_util.h>

class goto_coveraget
{
public:
  // add an assert(0)
  // - at the beginning of each GOTO program
  // - at the beginning of each branch body
  // - before each END_FUNCTION statement
  void add_false_asserts(goto_functionst &goto_functions);

  // insert an assert(0) as instrumentation BEFORE each instruction
  void
  insert_false_assert(goto_programt &goto_program, goto_programt::targett &it);

  // convert every assertion to an assert(0)
  void
  make_asserts_false(goto_functionst &goto_functions, const namespacet &ns);

  int get_total_instrument() const;

  void gen_assert_instance(goto_functionst goto_functions);
  int get_total_assert_instance() const;

protected:
  static int total_instrument;
  static int total_assert_instance;
};
