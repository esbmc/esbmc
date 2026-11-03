int main()
{
  int i = 0;
  int done = 1;
  /* The loop's exit is a break whose test the body computes; `done` is
     already 1 before the loop, yet the loop runs. The inductive step must
     not read that stale value as "the loop is never entered". */
  while (1)
  {
    done = (i == 5);
    if (done)
      break;
    i++;
  }
  __ESBMC_assert(i != 5, "the loop leaves with i == 5");
  return 0;
}
