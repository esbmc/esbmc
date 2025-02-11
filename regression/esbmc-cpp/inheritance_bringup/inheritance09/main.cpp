/*
 * Multiple inheritance: base pointer to derived class object
 */
#include <cassert>

class Base1 {
public:
   Base1( int parameterValue )
   {
      value = parameterValue;
   }

   int getData() const
   {
      return value;
   }

protected:
   int value;
};

class Base2
{
public:
   Base2( char characterData )
   {
      letter = characterData;
   }

   char getData() const
   {
      return letter;
   }
protected:
   char letter;
};

class Derived : public Base1, public Base2
{
public:
   Derived( int integer, char character, double double1 )
      : Base1( integer ), Base2( character ), real( double1 ) { }

   double getReal() const {
      return real;
   }

private:
   double real;
};


int main()
{
   Base1 base1( 10 ), *base1Ptr = 0;
   Base2 base2( 'Z' ), *base2Ptr = 0;
   Derived derived( 7, 'A', 3.5 );

   assert(base1.getData() == 10);
   assert(base2.getData() == 'Z');

   assert(derived.Base1::getData() == 7);
   assert(derived.Base2::getData() == 'A');
   assert(derived.getReal() == 3.5);

   base1Ptr = &derived;
   assert(base1Ptr->getData() == 7);

   base2Ptr = &derived;
   assert(base2Ptr->getData() == 'A');
   return 0;
}
