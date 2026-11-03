// probe's ports holds &x in an array element, which no pin can reach, but
// every call of probe declares ports afresh: the inductive step never reads
// its havoced value, so it needs no pin and the step proves count < 3.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int x;
int count;

void probe(void)
{
  int *ports[2] = {&x, 0};
  *ports[0] = 1;
  count = (count + 1) % 3;
}

int main()
{
  for (;;)
  {
    probe();
    __VERIFIER_assert(count < 3);
  }
}
