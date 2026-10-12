// init writes through termios, loaded through tty, which points to no object
// even after ext stores through it. The inductive step needs no havoc for that
// write and proves calls < 3.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct tty
{
  int *termios;
};

void ext(struct tty *);

int calls;

void init(struct tty *tty)
{
  ext(tty);
  int *termios = tty->termios;
  *termios = 0;
  calls = (calls + 1) % 3;
}

int main()
{
  struct tty *tty;
  for (;;)
  {
    init(tty);
    __VERIFIER_assert(calls < 3);
  }
}
