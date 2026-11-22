/* envp[0] .. envp[envp_size-1] point to strings, so reading them is not a
   dereference failure up to --argv-max-args (esbmc/esbmc#10382). */
int main(int argc, char **argv, char **envp)
{
  if (envp[0] && envp[1])
    return envp[0][0] + envp[1][0];
  return 0;
}
