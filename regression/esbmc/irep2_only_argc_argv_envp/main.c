/* The three-argument form: envp'/envp_size' come from the same side effect. */
int main(int argc, char **argv, char **envp)
{
  return argc > 0 ? 0 : (int)envp[0][0];
}
