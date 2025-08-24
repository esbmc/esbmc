void reach_error()
{
}
extern int nondet_int(void);

int main()
{
  int x = nondet_int();
  if (x < 2)
    reach_error();
  else if (x > 3)
    reach_error();
  else
    reach_error();
}
