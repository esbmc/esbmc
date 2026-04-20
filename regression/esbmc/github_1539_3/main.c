#include <assert.h>

/* Same int-to-ptr scenario as github_1539_2, but run under --smt-symex-assert
 * to exercise runtime_encoded_equationt::ask_solver_question().  That path
 * pushes two solver contexts and calls dec_solve() twice at the pushed level
 * before returning to the base solve.  pending_int_to_ptr_casts must be
 * re-emitted at the base level after the pushed frames are popped, otherwise
 * the int-to-ptr range constraint is lost and the counterexample is missed. */
int main()
{
  char *str0 = (char *)0x100;
  char *str1 = "";
  assert(str0 != str1);
}
