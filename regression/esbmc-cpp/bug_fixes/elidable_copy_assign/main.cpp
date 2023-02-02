class Foo {
  public:
    Foo() {};
    ~Foo() {};
    void Execute() {};
};

int main()
{
  auto foo = Foo();
  foo.Execute();
  return 0;
}
