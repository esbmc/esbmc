// A C module with its own main: it collides with the LD front end's
// __ESBMC_main and, with --secondary-entry-point, is kept as
// __ESBMC_secondary_main while the LD program's property is still checked.
int main(void)
{
  return 0;
}
