#pragma once
// The virtual method AND the constructor are only declared here; both are
// defined in shape.cpp — a different translation unit from main.cpp.  The vptr
// initialisation a constructor performs used to be driven by a flag that was
// only set while the *class* was being converted, so a constructor body
// converted in another TU emitted no vptr assignment.  Virtual dispatch then
// found no target ("dereference failure: invalid pointer").
class Shape
{
public:
  Shape(int id);
  virtual int kind() const; // base returns 1
  int id_;
};
class Circle : public Shape
{
public:
  Circle(int id);
  int kind() const override; // returns 2
};
