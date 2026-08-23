/* github_6212_nondet_extent_frame_pass:
 * Companion to github_6212_nondet_extent_frame_fail without the out-of-frame
 * write. The frame check now reads a parameter whose extent is nondet, so this
 * pins that it does not report the read it invents itself, on the paths where
 * the extent does not cover the pointee or where b turns out to alias r.
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
}

int main()
{
  return 0;
}
