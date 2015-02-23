// Ex. 4.18: ex04_18.cpp
// What does this program do?
#include <iostream>

using std::cout; 
using std::endl;

int whatIsThis( int [], int );  // function prototype

int main()
{
   const int arraySize = 10;
   int a[ arraySize ] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

   int result = whatIsThis( a, arraySize );

   cout << "Result is " << result << endl;

   return 0;  // indicates successful termination

} // end main

// What does this function do?
int whatIsThis( int b[], int size )
{
   // base case
   if ( size == 1 )
      return b[ 0 ];

   // recursive step
   else
      return b[ size - 1 ] + whatIsThis( b, size - 1 );

} // end function whatIsThis

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
