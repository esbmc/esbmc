/* A libc that implements strchr by forwarding to the builtin must not
   have that call rewritten into a call to itself. */
char *strchr(const char *s, int c)
{
  return __builtin_strchr(s, c);
}

int main(void)
{
  char s[3] = "ab";
  strchr(s, 'b');
  return 0;
}
