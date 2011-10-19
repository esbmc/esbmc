// Fig. 7.12: fig07_12.cpp 
// Non-friend/non-member functions cannot access
// private data of a class.
#include <iostream>

using std::cout;
using std::endl;

// Count class definition
class Count {

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

// function tries to modify private data of Count,
// but cannot because function is not a friend of Count
void cannotSetX( Count &c, int val )
{
   c.x = val;  // ERROR: cannot access private member in Count

} // end function cannotSetX

int main()
{
   Count counter;            // create Count object

   cannotSetX( counter, 3 ); // cannotSetX is not a friend

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
