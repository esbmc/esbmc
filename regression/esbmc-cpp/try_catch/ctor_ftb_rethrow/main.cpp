// C++ [except.handle]/15: a handler of a constructor's function-try-block
// cannot swallow an exception thrown while constructing a base/member
// subobject; at the end of the handler the exception is rethrown. So the
// exception here escapes main and terminates -> VERIFICATION FAILED.
struct Base
{
  Base() { throw 5; }
};
struct Derived : Base
{
  Derived() try : Base() {} catch (int) {}
};
int main()
{
  Derived d;
  return 0;
}
