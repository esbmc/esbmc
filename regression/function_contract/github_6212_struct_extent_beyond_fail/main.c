/* github_6212_struct_extent_beyond_fail:
 *   is_fresh states one S, so the extent it gives a struct parameter admits
 *   s[0] only and the access to s[3] must be caught.
 */
typedef struct
{
  int x;
} S;

void f(S *s)
{
  __ESBMC_requires(__ESBMC_is_fresh(s, sizeof(S)));
  __ESBMC_ensures(1);
  s[3].x = 1;
}

int main()
{
  return 0;
}
