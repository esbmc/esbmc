// The address reaches p inside a struct with no pointer member: the analysis
// has to carry it, or the loop's write to x goes unhavocked.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct s
{
  unsigned long a;
};

int x;

int main()
{
  struct s s1, s2;
  s1.a = (unsigned long)&x;
  s2 = s1;
  int *p = (int *)s2.a;
  int i = 0;
  for (;;)
  {
    p[0] = p[0] + 1;
    i++;
    __VERIFIER_assert(x < 10);
  }
}
