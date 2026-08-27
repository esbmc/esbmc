/* github_6212_struct_extent_stated_pass:
 * Same body as github_6212_struct_extent_unstated_fail with the extent
 * stated, which is the migration a contract needs once a struct pointer
 * parameter no longer carries an assumed one-element backing.
 */
typedef struct
{
  int x;
} S;

void f(S *s)
{
  __ESBMC_requires(s != 0);
  __ESBMC_requires(__ESBMC_is_fresh(s, sizeof(S)));
  __ESBMC_ensures(s->x == 1);
  s->x = 1;
}

int main()
{
  return 0;
}
