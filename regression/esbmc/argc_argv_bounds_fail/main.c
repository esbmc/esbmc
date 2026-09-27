/* One past argv's last valid index. The counterpart of argc_argv_bounds:
   without both halves neither pins the size (esbmc/esbmc#4715). */
int main(int argc, char **argv)
{
  return argv[argc + 1] == 0 ? 0 : 1;
}
