// #2175: --function main on an ESBMC goto binary keeps main's argc/argv.
int main(int argc, char **argv)
{
  __ESBMC_assert(argc >= 1, "argc is at least one");
  return 0;
}
