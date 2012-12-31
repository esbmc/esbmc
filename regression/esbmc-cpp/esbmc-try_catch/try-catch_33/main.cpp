class X {
  public:
    X() throw() { }
};

int main()
{
  try {
    X x;
    throw 5;
  }
  catch(int e) {
    return 1;
  }
  return 0;
}
