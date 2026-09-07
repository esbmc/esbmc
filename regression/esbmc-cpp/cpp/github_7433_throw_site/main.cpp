/* The uncaught-exception property is reported at the throw, not at the
   entry epilogue that synthesizes it. Here the throw is caught, so the
   property holds -- but it must still name line 11. */
struct my_error
{
};

void maybe_throw(int x)
{
  if (x > 0)
    throw my_error();
}

int nondet_int();

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
