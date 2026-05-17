/* Pins the v1 erase path of goto_loop_simplify: a for loop whose
 * induction variable is local-scoped (DEAD immediately after exit)
 * gets eliminated entirely. Verification produces zero VCCs because
 * the loop is gone before symex runs. */
int main()
{
  for (int i = 0; i < 17; i++)
    ;
  return 0;
}
