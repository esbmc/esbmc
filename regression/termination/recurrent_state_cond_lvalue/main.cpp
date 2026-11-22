// A conditional l-value names no single variable, so the check leaves a loop
// writing one alone. Taking its state as empty would see a repeat on the second
// visit.
int nondet_int();

int next(int v)
{
  return v + 1;
}

int main()
{
  int x = 0, y = 0;
  bool c = nondet_int() != 0;
  while (x < 3 && y < 3)
    (c ? x : y) = next(x + y);
  return 0;
}
