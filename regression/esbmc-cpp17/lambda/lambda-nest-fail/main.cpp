#include <cassert>

class TestClass
{
public:
  int a;
  int b;

  TestClass(int a, int b) : a(a), b(b)
  {
  }

  void testLambdaCaptureNested()
  {
    int x = 10, y = 20;
    auto outerLambda = [x, &y, *this]()
    {
      auto innerLambda = [x, &y, *this]()
      {
        assert(x == 10);
        assert(y == 20);
        assert(this->a == 10);
        assert(this->b == 20);
      };
      innerLambda();
    };
    outerLambda();
  }

  void testLambdaMixedCapture()
  {
    int x = 10, y = 20, z = 30;
    auto outerLambda = [x, &y, z]()
    {
      y += 5;
      auto innerLambda = [x, &y, z]()
      {
        assert(x == 10);
        assert(y == 25);
        assert(z == 30);
      };
      innerLambda();
    };
    outerLambda();
    assert(y == 20);  //should change to 25
  }

  void testLambdaCaptureThisByReference()
  {
    auto outerLambda = [this]()
    {
      this->a += 5;
      this->b += 5;
      auto innerLambda = [this]()
      {
        assert(this->a == 15);
        assert(this->b == 25);
      };
      innerLambda();
    };
    outerLambda();
  }
};

int main()
{
  TestClass obj(10, 20);

  obj.testLambdaMixedCapture();
  obj.testLambdaCaptureNested();
  obj.testLambdaCaptureThisByReference();
  return 0;
}