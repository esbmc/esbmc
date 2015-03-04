class X {
  public:
    X() throw() { throw 5; }
};

int main()
{
  try {
    X x;
  }
  catch(int e) {
    return 1;
  }
  return 0;
}
