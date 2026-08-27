/* github_6212_struct_extent_beyond_fail:
 * Companion to github_6212_struct_extent_unstated_fail, one element further
 * out. s[3] was caught even under the old one-element stack backing, so this
 * pins that the access past the first element stays caught now that the
 * extent is nondet, rather than only the first element changing verdict.
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
