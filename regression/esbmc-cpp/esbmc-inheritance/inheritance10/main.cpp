#include <cassert>
#include <iostream>
using std::ostream;
using std::cout;
using std::endl;

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

	void setSecond(){
		second_value = 10;
	}

	int getSecond(){
		return second_value;
	}
protected:
   int value;
	int second_value;
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
    void display()
    {
       cout << "    Integer: " << value << "\n  Character: "
          << letter << "\nReal number: " << real;

    }

private:
   double real;
};


int main()
{
   Base1 base1( 10 ), *base1Ptr = 0;
   Base2 base2( 'Z' ), *base2Ptr = 0;
   Derived derived( 7, 'A', 3.5 );

   cout << base1.getData()
        << base2.getData();
   derived.display();

	derived.setSecond();
	assert(derived.getSecond() == 20);

   cout << derived.Base1::getData()
        << derived.Base2::getData()
        << derived.getReal() << "\n\n";

   base1Ptr = &derived;
   cout << base1Ptr->getData() << '\n';

   base2Ptr = &derived;
   cout << base2Ptr->getData() << endl;
   return 0;
}
