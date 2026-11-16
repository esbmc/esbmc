// In every state an input can reach, an arm fires for every input, so no
// input leaves the state alone; inputs 1 and 2 still alternate a between
// 0 and 1 forever, a cycle of two iterations from the initial state.

int a = 0;

extern int __VERIFIER_nondet_int(void);
extern void exit(int);

int update(int input)
{
  if (a == 0 && input == 1)
  {
    a = 1;
    return 1;
  }
  if (a == 1 && input == 2)
  {
    a = 0;
    return 2;
  }
  if (a == 0 || a == 1)
    exit(0);
  if (a != 0)
    exit(0);
  return -2;
}

int main(void)
{
  int sink = 0;
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1 && input != 2)
      return -2;
    sink = update(input);
  }
  return sink;
}
