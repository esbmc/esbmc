#include<cassert>

class X
{
  public:
    X() { throw 5; }
};

class Y : public X
{
  public:
    Y() try : X() { 

    } catch(int) { }
};

int main()
{
  Y y;
  return 0;
}
