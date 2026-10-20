/* #8097: as github_8097_trace_index_through_pointer_fail, with a member of
 * the element as the assigned lvalue. */
typedef struct
{
  unsigned x;
} E;

typedef struct
{
  unsigned a[2];
  E c[2];
} S;

unsigned nondet_uint(void);

void f(S *s)
{
  for (int i = 0; i < 2; i++)
  {
    s->a[i] = nondet_uint() % 2;
    s->c[s->a[i]].x = i + 1;
  }
}

int main(void)
{
  S s = {{0, 0}, {{0}, {0}}};
  f(&s);
  __ESBMC_assert(s.c[0].x + s.c[1].x != 3, "c holds 1 and 2");
  return 0;
}
