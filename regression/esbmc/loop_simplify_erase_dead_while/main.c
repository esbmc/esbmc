/* Same v1 erase contract as loop_simplify_erase_dead_for but for
 * the while loop shape. The frontend lowers `while` and `for` to the
 * same IF/back-edge structure, so the pass should handle them
 * uniformly. */
int main()
{
  {
    int i = 0;
    while (i < 17)
      i++;
    /* end of scope — DEAD i immediately follows */
  }
  return 0;
}
