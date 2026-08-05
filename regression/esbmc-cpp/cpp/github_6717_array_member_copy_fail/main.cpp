// Negative counterpart of github_6717_array_member_copy: the copy carries the
// real element values, so a claim contradicting one is refuted rather than
// vacuously held (issue #6717).
struct A
{
  int i;
};

struct B
{
  A ar[3];
};

int main()
{
  B b;
  b.ar[0].i = 7;
  B b2(b);
  __ESBMC_assert(b2.ar[0].i == 8, "must not hold");
  return 0;
}
