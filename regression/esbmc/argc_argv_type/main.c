/* declare_argc_argv reads main's type off the IREP2 side, so on the default
   path -- where clang_c_convert wrote that type legacy -- the symbol's lazy
   forward migration has to preserve the argument types (esbmc/esbmc#4715). */
int main(int argc, char **argv)
{
  return argc > 0 ? 0 : (int)argv[0][0];
}
