// The loop's callee writes a conditional l-value, which names no single
// variable, so the check leaves the loop alone.
int nondet_int();

int x, y;
bool c;

void step()
{
  (c ? x : y) = x + y + 1;
}

int main()
{
  c = nondet_int() != 0;
  while (x < 3 && y < 3)
    step();
  return 0;
}
