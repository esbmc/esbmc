/* github_6484_frame_symbolic_extent_fail:
 * The natural contract for a buffer-with-length API: the extent is the
 * parameter n, so n == 0 makes buf[0] invalid and this must FAIL.
 *
 * Phase 2B bounds a nondet witness index by extent/sizeof(elem). Emitting that
 * bound as an ASSUME would force n >= 1, which is the #6212 assumption wearing
 * a frame clause; for a zero extent it is ASSUME(false), which discharges the
 * whole wrapper vacuously. The bound must be a clamp, not an assumption.
 */
void f(char *buf, unsigned long n)
{
  __ESBMC_requires(__ESBMC_is_fresh(buf, n));
  __ESBMC_assigns(buf[0]);
  __ESBMC_ensures(1);
  buf[0] = 1;
}

int main()
{
  return 0;
}
