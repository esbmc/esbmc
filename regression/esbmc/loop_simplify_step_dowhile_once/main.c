/* Pins the at-least-one-iteration semantics of do-while.
 *
 * The init i = 20 already violates the continuation condition i < 17,
 * so a for/while form would not execute the body at all and exit with
 * i = 20. But do-while always runs the body once first: i becomes 21,
 * the back-edge `21 < 17` is false, and the loop exits with i = 21.
 *
 * Step recognition models this by stepping init by step before applying
 * the standard formula. Without that adjustment we'd predict i = 20
 * (incorrect — the loop would still appear empty under for/while
 * semantics). */
int main()
{
  int i = 20;
  do {
    i++;
  } while (i < 17);
  __ESBMC_assert(i == 21, "do-while runs body once before testing cond");
  return 0;
}
