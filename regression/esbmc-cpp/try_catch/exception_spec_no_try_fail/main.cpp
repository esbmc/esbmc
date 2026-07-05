// An exception specification is a function-boundary contract, enforced even
// when there is no surrounding try/catch region at all. This is the case the
// old THROW_DECL representation silently ignored, because it attached the
// specification to the nearest active catch region (here, none).
void f() noexcept
{
  throw 5;
}

int main()
{
  f();
  return 0;
}
