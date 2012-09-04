// Fig. 22.13 : fig22_13.cpp
// Demonstrating operators .* and ->*.
#include <iostream>

using std::cout;
using std::endl;

// class Test definition
class Test {
public:
   void function() { cout << "function\n"; }
   int value;  // public data member
};  // end class Test

void arrowStar( Test * );
void dotStar( Test * );

// access member function of Test object using ->*
void arrowStar( Test *testPtr )
{
   // declare function pointer
   void ( Test::*memPtr )() = &Test::function;

   // invoke function indirectly
   ( testPtr->*memPtr )(); 
   
}  // end arrowStar

// access members of Test object data member using .*
void dotStar( Test *testPtr2 )
{
   int Test::*vPtr = &Test::value;   // declare pointer

   cout << ( *testPtr2 ).*vPtr << endl;  // access value
   
}  // end dotStar

int main()
{
   Test test;    
  
   test.value = 8;      // assign value 8 
   arrowStar( &test );  // pass address to arrowStar
   dotStar( &test );    // pass address to dotStar

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
