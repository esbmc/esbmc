int idx(int i)
{
  return i;
}

int main(void)
{
  int a[8][8];

  /* A propagated 2-D array reaches symex as a DAG, not a tree; the walks over
     it must memoise or this does not terminate. See R52 in
     docs/roadmap/goto-symex-verification-plan.md. */
  a[0][0] = 0;
  a[0][1] = 1;
  a[0][2] = 2;
  a[0][3] = 3;
  a[0][4] = 4;
  a[0][5] = 5;
  a[0][6] = 6;
  a[0][7] = 7;
  a[1][0] = 8;
  a[1][1] = 9;
  a[1][2] = 10;
  a[1][3] = 11;
  a[1][4] = 12;
  a[1][5] = 13;
  a[1][6] = 14;
  a[1][7] = 15;
  a[2][0] = 16;
  a[2][1] = 17;
  a[2][2] = 18;
  a[2][3] = 19;
  a[2][4] = 20;
  a[2][5] = 21;
  a[2][6] = 22;
  a[2][7] = 23;
  a[3][0] = 24;
  a[3][1] = 25;
  a[3][2] = 26;
  a[3][3] = 27;
  a[3][4] = 28;
  a[3][5] = 29;
  a[3][6] = 30;
  a[3][7] = 31;

  __ESBMC_assert(
    a[idx(3)][idx(7)] == 31, "the last store is visible through the chain");
  return 0;
}
