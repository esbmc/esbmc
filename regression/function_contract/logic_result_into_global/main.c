/* A logic result stored into a global is read in another function: the call
   must be lowered, not dropped as unread. */
__ESBMC_logic int sq(int x) { return x * x; }
int g;
void set(void) { g = sq(3); }
int main()
{
  set();
  __ESBMC_assert(g == 9, "stored");
  return 0;
}
