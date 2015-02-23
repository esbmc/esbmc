// Fig. 7.4: fig07_04.cpp
// Using a member initializer to initialize a
// constant of a built-in data type.
#include <iostream>

using std::cout;
using std::endl;

class Increment {

public:
   Increment( int c = 0, int i = 1 );  // default constructor

   void addIncrement() 
   { 
      count += increment; 

   } // end function addIncrement

   void print() const;     // prints count and increment

private:
   int count;
   const int increment;    // const data member

}; // end class Increment

// constructor 
Increment::Increment( int c, int i )
   : count( c ),     // initializer for non-const member
     increment( i )  // required initializer for const member     
{ 
   // empty body

} // end constructor Increment

// print count and increment values
void Increment::print() const
{
   cout << "count = " << count
        << ", increment = " << increment << endl;

} // end function print

int main()
{
   Increment value( 10, 5 );

   cout << "Before incrementing: ";
   value.print();

   for ( int j = 0; j < 3; j++ ) {
      value.addIncrement();
      cout << "After increment " << j + 1 << ": ";
      value.print();
   }

   return 0;

} // end main

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
