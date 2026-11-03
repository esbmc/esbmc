#include <cassert>

struct Base
{
  virtual int &forward(int &value) const noexcept
  {
    int &first = update(value);
    int &second = update(value);
    int &third = update(value);
    assert(&first == &second);
    assert(&second == &third);
    return third;
  }

  virtual int &update(int &value) const noexcept
  {
    ++value;
    return value;
  }
};

struct Derived : Base
{
  int &update(int &value) const noexcept override
  {
    value += 2;
    return value;
  }
};

struct Outer
{
  struct Parameter
  {
    int value;

    virtual int call(Outer &outer)
    {
      return outer.read(*this);
    }
  };

  virtual int read(Parameter);
};

int Outer::read(Parameter renamed)
{
  return renamed.value + 4;
}

int main()
{
  Outer outer;
  Outer::Parameter parameter;
  parameter.value = 3;
  assert(parameter.call(outer) == 7);

  Base concrete;
  int base_value = 1;
  int &base_result = concrete.forward(base_value);
  assert(&base_result == &base_value);
  assert(base_value == 4);

  Derived object;
  const Base *base = &object;
  int value = 1;
  int &result = base->forward(value);
  assert(&result == &value);
  assert(value != 7);
}
