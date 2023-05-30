#include <goto-programs/goto_functions.h>

// convert every assertion to an assert(0)
void add_false_assert(goto_functionst &goto_functions);

// add an assert(0) before each RETURN/END_FUNCTION statement
void make_assert_false(goto_functionst &goto_functions);
