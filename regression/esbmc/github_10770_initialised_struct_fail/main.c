extern int __VERIFIER_nondet_int(void);
extern void reach_error(void);

struct s
{
  int f;
  int g;
};

int main(void)
{
  struct s t = {0};
  t.f = __VERIFIER_nondet_int();
  if (t.f == 7)
    reach_error();
  return 0;
}
