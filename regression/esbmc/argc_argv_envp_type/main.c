/* The three-argument main on the default adjuster path, where clang_c_convert
   wrote main's type legacy and declare_argc_argv reads it back through the
   symbol's lazy forward migration (esbmc/esbmc#4715). */
int main(int argc, char **argv, char **envp)
{
  return argc > 0 && envp[0] != 0 ? 0 : (int)argv[0][0];
}
