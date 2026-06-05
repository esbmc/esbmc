void reach_error(void) {}

static void check(int x)
{
  if (x < 0)
    reach_error();
}

int main(void)
{
  check(-1);
  return 0;
}
