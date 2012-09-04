// Fig. 3.14: fig03_14.cpp
// Recursive factorial function.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

unsigned long factorial( unsigned long ); // function prototype

int main()
{
   // Loop 10 times. During each iteration, calculate 
   // factorial( i ) and display result.
   for ( int i = 0; i <= 10; i++ )
      cout << setw( 2 ) << i << "! = " 
           << factorial( i ) << endl;

   return 0;  // indicates successful termination

} // end main

// recursive definition of function factorial
unsigned long factorial( unsigned long number )
{
   // base case
   if ( number <= 1 )  
      return 1;

   // recursive step
   else                
      return number * factorial( number - 1 );

} // end function factorial

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
