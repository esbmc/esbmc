// Negative counterpart of builtin_string_rewrites: the rewritten builtin
// returns the real result, so a claim contradicting it is refuted.
int main(void)
{
  char s[6] = "hello";
  __ESBMC_assert(__builtin_strchr(s, 'l') == s + 3, "must not hold");
  return 0;
}
