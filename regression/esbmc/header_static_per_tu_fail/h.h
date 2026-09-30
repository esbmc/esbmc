/* Each file that includes this header has its own copy of both. */
static int g;
static int counter(void)
{
  static int c;
  ++g;
  return ++c;
}
