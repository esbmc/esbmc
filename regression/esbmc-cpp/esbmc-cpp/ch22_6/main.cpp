// Fig. 22.1: fig22_01.cpp
// Demonstrating operator const_cast. 
#include <iostream>

using std::cout;
using std::endl;
      
// class ConstCastTest definition
class ConstCastTest {
public:
   void setNumber( int );
   int getNumber() const;
   void printNumber() const;
private:
   int number;
};  // end class ConstCastTest

// set number
void ConstCastTest::setNumber( int num ) { number = num; }

// return number
int ConstCastTest::getNumber() const { return number; }

// output number
void ConstCastTest::printNumber() const
{
   cout << "\nNumber after modification: ";
	
   // cast away const-ness to allow modification
   const_cast< ConstCastTest * >( this )->number--;

   cout << number << endl;

}  // end printNumber

int main()
{
   ConstCastTest test;   // create ConstCastTest instance

   test.setNumber( 8 );  // set private data number to 8
   
   cout << "Initial value of number: " << test.getNumber();

   test.printNumber();
   return 0;

}  // end main

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/
