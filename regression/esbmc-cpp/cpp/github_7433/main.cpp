struct my_error
{
};

int nondet_int();

void maybe_throw(int x)
{
  if (x > 0)
    throw my_error();
}

int main()
{
  try
  {
    maybe_throw(nondet_int());
  }
  catch (const my_error &)
  {
  }
}
