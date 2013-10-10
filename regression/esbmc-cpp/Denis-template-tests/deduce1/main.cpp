#include <cassert>
#include <cstddef>
// { dg-do run }

// Copyright (C) 2002 Free Software Foundation, Inc.
// Contributed by Nathan Sidwell 13 Sep 2002 <nathan@codesourcery.com>

template <typename T> int Foo (T const *)
{
  return 1;
}
template <typename T> int Foo (T const &)
{
  return 2;
}
template <typename T, size_t I> int Foo (T const (&ref)[I])
{
  return 0;
}

int main ()
{
  static int array[4] = {};
  
  assert(0 == ( Foo (array)));
}

