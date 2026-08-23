/* github_6212_nondet_extent_frame_fail:
 * A parameter with a nondet extent still gets the assigns-clause frame check.
 * The check used to be skipped for those parameters, leaving the write to b->x
 * to the bounds check alone -- so turning bounds checking off verified an
 * out-of-frame write. The snapshot and its assertion are now taken under the
 * condition that the extent covers the pointee, which is sound whether or not
 * bounds checking is on.
 */
typedef struct
{
  int x;
} S;

void f(S *r, S *b)
{
  __ESBMC_requires(r != 0 && b != 0);
  __ESBMC_assigns(r->x);
  __ESBMC_ensures(1);
  r->x = 1;
  b->x = 2; /* illegal: b->x is not in the assigns clause */
}

int main()
{
  return 0;
}
