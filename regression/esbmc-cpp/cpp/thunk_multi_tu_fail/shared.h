#pragma once
// A polymorphic class with a virtual destructor and an overriding method.
// When this header is included by more than one translation unit, ESBMC used
// to generate the override's thunk (and its argument symbols) once per TU and
// abort with "Failed to add arg symbol ... already exists"; the follow-up
// struct-copy in value_set then tripped a member2t component-lookup assert on
// the class-specific vtable-pointer member.
struct Base
{
  virtual ~Base() {}
  virtual int f() { return 1; }
};
struct Derived : Base
{
  int f() override { return 2; }
};
int use_a();
int use_b();
