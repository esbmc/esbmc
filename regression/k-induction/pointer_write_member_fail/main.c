// The loop writes s.a only through q->a; the inductive step must havoc s.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct s
{
  int a;
  int b;
};

int main()
{
  struct s s = {0, 0};
  struct s *q = &s;
  int i = 0;
  for (;;)
  {
    q->a++;
    i++;
    __VERIFIER_assert(s.a < 10);
  }
}
