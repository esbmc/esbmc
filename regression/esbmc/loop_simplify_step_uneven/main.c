/* Step recognition with a step that does NOT divide (bound - init).
 * for(i=0; i<17; i+=3) iterates with i in {0, 3, 6, 9, 12, 15} and
 * exits at i = 18 — the first value satisfying !(i < 17). Pins the
 * ceiling-division branch of compute_post_value (not a simple
 * `i = bound` shortcut). */
int main()
{
  int i;
  for (i = 0; i < 17; i += 3)
    ;
  __ESBMC_assert(i == 18, "ceiling division gives 18, not 17");
  return 0;
}
