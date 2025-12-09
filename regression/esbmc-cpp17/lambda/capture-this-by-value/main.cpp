#include <cassert>

class TestClass
{
public:
  int a;
  int b;

  TestClass(int a, int b) : a(a), b(b)
  {
  }

  void testLambdaCaptureThisByValue()
  {
    auto lambda = [*this]() {
      assert(this->a == 10);
      assert(this->b == 20);
    };
    lambda();
  }

  void testLambdaCaptureThisByValueModify()
  {
    auto lambda = [*this]() mutable {
      this->a = 30;
      this->b = 40;
      assert(this->a == 30);
      assert(this->b == 40);
    };
    lambda();
    assert(a == 10); // Original object should remain unchanged
    assert(b == 20);
  }

  void testLambdaCaptureThisByValueConst() const
  {
    auto lambda = [*this]() {
      assert(this->a == 10);
      assert(this->b == 20);
    };
    lambda();
  }

  void testLambdaCaptureThisByValueNested()
  {
    auto outerLambda = [*this]() {
      auto innerLambda = [*this]() {
        assert(this->a == 10);
        assert(this->b == 20);
      };
      innerLambda();
    };
    outerLambda();
  }
};

int main()
{
  TestClass obj(10, 20);

  obj.testLambdaCaptureThisByValue();
  obj.testLambdaCaptureThisByValueModify();
  obj.testLambdaCaptureThisByValueConst();
  obj.testLambdaCaptureThisByValueNested();
  return 0;
}