// Fig. 7.11: fig07_11.cpp  
// Friends can access private members of a class.
#include <iostream>

using std::cout;
using std::endl;

// Count class definition 
// (note that there is no friendship declaration)
class Count {
   friend void setX( Count &, int ); // friend declaration

public:

   // constructor
   Count()
      : x( 0 )  // initialize x to 0
   { 
      // empty body
   
   } // end constructor Count

   // output x 
   void print() const       
   { 
      cout << x << endl; 

   } // end function print

private:
   int x;  // data member

}; // end class Count

// function setX can modify private data of Count 
// because setX is declared as a friend of Count  
void setX( Count &c, int val )
{
   c.x = val;  // legal: setX is a friend of Count

} // end function setX

int main()
{
   Count counter;       // create Count object

   cout << "counter.x after instantiation: ";
   counter.print();

   setX( counter, 8 );  // set x with a friend

   cout << "counter.x after call to setX friend function: ";
   counter.print();

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
