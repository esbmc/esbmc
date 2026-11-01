/* A logic result stored into an array element is read later: the call must
   be lowered, not dropped as unread. */
__ESBMC_logic int sq(int x) { return x * x; }
int main()
{
  int a[1] = {0};
  a[0] = sq(3);
  __ESBMC_assert(a[0] == 0, "false: a[0] is 9");
  return 0;
}
