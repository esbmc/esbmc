/* github_6483_struct_extent_beyond_fail:
 *   Companion to github_6483_struct_extent_knownbug. The single-element stack
 *   backing given to a struct pointer parameter admits only s[0], so an access
 *   to s[3] is not justified by the contract and is caught.
 */
typedef struct
{
  int x;
} S;

void f(S *s)
{
  __ESBMC_requires(s != 0);
  __ESBMC_ensures(1);
  s[3].x = 1;
}

int main()
{
  return 0;
}
