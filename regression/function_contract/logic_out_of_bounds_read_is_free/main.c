/* A definition reading past its object yields a free value, not a failure,
   and still one value per argument. */
__ESBMC_logic int at(const char *p, int i) { return p[i]; }
int main()
{
  char c = 1;
  __ESBMC_assert(at(&c, 0) == 1, "in bounds");
  __ESBMC_assert(at(&c, 5) == at(&c, 5), "congruent");
  return 0;
}
