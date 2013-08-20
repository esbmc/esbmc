// Exercise 3.2: ex03_02.cpp
#include <iostream>

using std::cout;
using std::endl;

int cube( int y );  // function prototype

int main()
{
   int x;

   // loop 10 times, calculate cube of x and output results
   for ( x = 1; x <= 10; x++ )
      cout << cube( x ) << endl;

   return 0;  // indicates successful termination

} // end main

// definition of function cube
int cube( int y )
{
   return y * y * y;
}


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
