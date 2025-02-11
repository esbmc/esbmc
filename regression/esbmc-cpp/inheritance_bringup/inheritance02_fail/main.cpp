/*
 * Single virtual inheritance: call base method in derived class
 */
#include <cassert>

class exforsys
{
public:
    exforsys(void) { x=0; }
    void f(int n1)
    {
      x= n1*5;
    }

    void add_one(void) { ++x; }
    int getX() { return x; }
private:
    int x;
};

class sample: virtual public exforsys
{
public:
    sample(void) { s1=0; }

    void f1(int n1)
    {
      s1=n1*10;
    }

    void add_one(void)
    {
      exforsys::add_one();
      ++s1;
    }
    int getS1() { return s1; }

private:
    int s1;
};

int main(void)
{
    sample s;
    s.f(10);
    assert(s.getX() == 50); // PASS
    s.f1(20);
    assert(s.getS1() == 200); // PASS
    s.add_one();
    assert(s.getX() == 51); // PASS
    assert(s.getS1() == 200); // FAIL, should be 201
}
