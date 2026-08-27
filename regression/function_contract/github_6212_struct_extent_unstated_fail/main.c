/* github_6212_struct_extent_unstated_fail:
 * The struct half of github_6212_unstated_extent_fail. A struct pointer
 * parameter used to keep a one-element stack backing, so s->x was admitted
 * even though the contract states no extent for s and a caller is free to
 * pass a pointer to less than one S. It now gets the same nondet extent as
 * any other pointer parameter, so the write must be caught.
 */
typedef struct
{
  int x;
} S;

void f(S *s)
{
  __ESBMC_requires(s != 0);
  __ESBMC_ensures(1);
  s->x = 1;
}

int main()
{
  return 0;
}
