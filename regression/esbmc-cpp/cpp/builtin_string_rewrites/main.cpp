// The __builtin_ spellings of <string.h> functions must reach the C library
// models from C++ too; unmodelled, each call returned nondet.
int main()
{
  char s[6] = "hello";
  __ESBMC_assert(__builtin_strchr(s, 'l') == s + 2, "strchr");
  __ESBMC_assert(__builtin_memchr(s, 'o', 5) == s + 4, "memchr");
  __ESBMC_assert(__builtin_strstr(s, "lo") == s + 3, "strstr");
  __ESBMC_assert(__builtin_strrchr(s, 'l') == s + 3, "strrchr");
  return 0;
}
