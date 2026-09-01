/* Negative twin: nothing catches the throw, so the property is violated --
   and it must still be reported at the raise site, line 11, not at the
   entry epilogue. */
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
  maybe_throw(nondet_int());
}
