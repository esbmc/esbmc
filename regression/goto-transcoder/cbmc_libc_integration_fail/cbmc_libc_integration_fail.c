extern unsigned long strlen(const char *);
extern int strncmp(const char *, const char *, unsigned long);
extern int isalpha(int);
extern int isdigit(int);
extern int toupper(int);
extern int atoi(const char *);

/* Validate a token of the form "<alpha-key>=<digits>" and return the integer
   value, or -1 on malformed input. Exercises the ctype and stdlib bridges plus
   object alignment on one realistic control-flow-heavy routine. */
int parse_kv(const char *s)
{
  unsigned long n = strlen(s);
  if (n < 3)
    return -1;
  unsigned long i = 0;
  if (!isalpha((int)s[i]))
    return -1;
  while (i < n && s[i] != '=')
  {
    if (!isalpha((int)s[i]))
      return -1;
    i++;
  }
  if (i == 0 || i >= n || s[i] != '=')
    return -1;
  i++;
  if (i >= n)
    return -1;
  for (unsigned long j = i; j < n; j++)
    if (!isdigit((int)s[j]))
      return -1;
  return atoi(s + i);
}

int main(void)
{
  __CPROVER_assert(parse_kv("ab=42") == 43, "well-formed key=value");
  __CPROVER_assert(parse_kv("x=7") == 7, "short well-formed");
  __CPROVER_assert(parse_kv("no_eq") == -1, "no equals sign");
  __CPROVER_assert(parse_kv("=5") == -1, "empty key");
  __CPROVER_assert(parse_kv("k=1a") == -1, "non-digit value");
  __CPROVER_assert(strncmp("abc", "abd", 2) == 0, "prefix compare");
  __CPROVER_assert(toupper((int)'a') == 'A', "case map");
  return 0;
}
