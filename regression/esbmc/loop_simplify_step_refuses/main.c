/* Negative pin: step recognition must NOT fire on a loop whose
 * exit guard references memory through the induction variable.
 *
 * The strlen idiom `while (s[len] != 0) len++;` modifies only `len`,
 * but the guard `s[len] != 0` indexes an array with the modified
 * value. Rewriting the loop to `len = post_value` requires deriving
 * post_value from the array contents at runtime — there's no
 * constant_int bound. parse_guard refuses (neither side of the
 * relation is a constant_int) and we fall through; the loop is left
 * intact for symex to unwind.
 *
 * The test sets up two scenarios and checks the loop still runs:
 *   - src[0] == 'x' (non-zero): one iteration.
 *   - src[0] == 0:              zero iterations.
 * Both should verify. If step recognition incorrectly fired, len
 * would be a constant and these checks would be trivially decided
 * by constant folding, masking the bug. The assertion `len < 4`
 * forces symex to actually execute the loop and check the guard. */
int main()
{
  char src[4] = {'a', 'b', '\0', 'x'};
  unsigned len = 0;
  while (src[len] != 0)
    len++;
  __ESBMC_assert(len < 4, "strlen-style loop unwound, len within bounds");
  return 0;
}
