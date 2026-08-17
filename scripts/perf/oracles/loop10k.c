/* Oracle for the #6831 W0 bisect: a task in the 10-100 s band, where a
 * multiplicative term is visible and the fixed library cost is noise.
 *
 *   esbmc loop10k.c --unwind 10000 --overflow-check --quiet
 *
 * On the plan's host that is ~11 s wall, spread across every phase rather than
 * hidden in one: 0.3 s GOTO creation, 0.8 s symex, 0.2 s caching, 0.7 s
 * slicing, 4.3 s encoding, 2.1 s solving. Nothing here is nondeterministic, so
 * the VCC and symex-assignment counts (79,992 and 120,003) are a fingerprint:
 * two builds that disagree on them differ in the work they do, not in how fast
 * they do it.
 *
 * --quiet is not cosmetic. The 10,000 unwinding lines are ~8 % of wall, and
 * common-mode work dilutes the ratio being measured.
 *
 * Deliberately not a nondet-indexed array: that shape lands almost all of its
 * time in the solver, which makes it a solver benchmark rather than an ESBMC
 * one. */
int main(void)
{
  int sum = 0;
  int prod = 1;

  for (int i = 1; i < 10000; i++)
  {
    sum += i * 3 + 1;
    prod += sum - i;
    sum -= prod / 2;
  }

  return sum;
}
