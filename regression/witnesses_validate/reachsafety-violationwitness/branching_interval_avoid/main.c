extern void reach_error(void);
void check(int cond)
{
  if (!cond)
  {
    reach_error();
  }
}
int main(void)
{
  check(0);
}
