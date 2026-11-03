/* github_6212_struct_extent_fail:
 *   The contract states only that s is non-null, so the write to s->x is not
 *   justified and must be caught. Struct parameters used to keep a
 *   one-element stack backing, which admitted it (#6212).
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
