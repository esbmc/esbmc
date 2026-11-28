// The __builtin_ spellings of <string.h> functions must reach the library
// models. Freestanding libcs (e.g. Trusted Firmware-A's) #define strchr and
// friends to __builtin_strchr; unmodelled, each call returned nondet.
int main(void)
{
  char s[6] = "hello";
  __ESBMC_assert(__builtin_memchr(s, 'l', 5) == s + 2, "memchr");
  __ESBMC_assert(__builtin_strchr(s, 'e') == s + 1, "strchr");
  __ESBMC_assert(__builtin_strrchr(s, 'l') == s + 3, "strrchr");
  __ESBMC_assert(__builtin_strncmp(s, "help", 3) == 0, "strncmp");
  __ESBMC_assert(__builtin_strstr(s, "ll") == s + 2, "strstr");
  __ESBMC_assert(__builtin_strspn(s, "eh") == 2, "strspn");
  __ESBMC_assert(__builtin_strcspn(s, "lo") == 2, "strcspn");
  __ESBMC_assert(__builtin_strpbrk(s, "ol") == s + 2, "strpbrk");

  char d[8] = "ab";
  __ESBMC_assert(__builtin_strcat(d, "cd") == d && d[3] == 'd', "strcat");
  __ESBMC_assert(__builtin_strncat(d, "efg", 2) == d && d[5] == 'f', "strncat");
  __ESBMC_assert(d[6] == '\0', "strncat terminates");

  char e[4];
  __ESBMC_assert(__builtin_stpcpy(e, "xy") == e + 2 && e[1] == 'y', "stpcpy");

  char *dup = __builtin_strdup("hi");
  __ESBMC_assert(!dup || (dup[1] == 'i' && dup[2] == '\0'), "strdup");
  return 0;
}
