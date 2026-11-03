// tty may also point to dev, whose termios points to state, so init's write
// reaches state. Unless the inductive step havocs state, it keeps its pre-loop
// 0 and the step proves state < 5.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
_Bool __VERIFIER_nondet_bool(void);

struct tty
{
  int *termios;
};

int state;
struct tty dev = {&state};

void init(struct tty *tty)
{
  int *termios = tty->termios;
  *termios = *termios + 1;
}

int main()
{
  struct tty *unset;
  struct tty *tty = __VERIFIER_nondet_bool() ? &dev : unset;
  for (;;)
  {
    init(tty);
    __VERIFIER_assert(state < 5);
  }
}
