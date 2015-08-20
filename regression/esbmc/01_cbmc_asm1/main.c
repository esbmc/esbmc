// this is a GCC extension
char *strerror(int) __asm("_" "strerror" "$UNIX2003");

int main()
{
  __asm("mov ax, dx");
}
