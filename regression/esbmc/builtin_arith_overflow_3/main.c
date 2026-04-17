typedef unsigned long ulong;
extern ulong nondet_ulong();

int main()
{
  ulong content_len = nondet_ulong();
  __builtin_umull_overflow(content_len, 10UL, &content_len);
}