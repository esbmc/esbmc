#include <cassert>
// { dg-do run }

// Copyright (C) 2002 Free Software Foundation, Inc.
// Contributed by Nathan Sidwell 14 Sep 2002 <nathan@codesourcery.com>

// PR 7768 template dtor pretty function wrong
//
// ESBMC uses the Clang frontend, whose __PRETTY_FUNCTION__ format
// ("X<void>::X() [T = void]") differs from GCC's
// ("X<T>::X() [with T = void]"). The implementation-defined wording is
// allowed to differ across compilers; this regression asserts the form
// ESBMC actually produces.

#include <string.h>

static bool ctor_ok = false;
static bool dtor_ok = false;

template <typename T>
struct X
{
  X()  { ctor_ok = (strcmp(__PRETTY_FUNCTION__, "X<void>::X() [T = void]") == 0); }
  ~X() { dtor_ok = (strcmp(__PRETTY_FUNCTION__, "X<void>::~X() [T = void]") == 0); }
};

int main()
{
  {
    X<void> x;
    assert(ctor_ok);
  }
  assert(dtor_ok);
  return 0;
}
