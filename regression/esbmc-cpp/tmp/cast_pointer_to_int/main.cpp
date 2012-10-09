class Test {
  public:
    Test(char *p)
    {
      ptr = p;
    }
    char *ptr;
};

int main()
{
  char *tmp, tmp2='a';
  unsigned int addr;
  tmp = &tmp2;
  Test test(tmp);
  addr = (unsigned int) test.ptr;
}
