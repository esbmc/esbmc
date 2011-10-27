// Fig. 5.6: fig05_06.cpp
// Cube a variable using pass-by-value.
#include <iostream>

using std::cout;
using std::endl;

int cubeByValue( int );   // prototype

int main()
{
   int number = 5;

   cout << "The original value of number is " << number;

   // pass number by value to cubeByValue
   number = cubeByValue( number );

   cout << "\nThe new value of number is " << number << endl;

   return 0;  // indicates successful termination

} // end main

// calculate and return cube of integer argument
int cubeByValue( int n )
{
   return n * n * n; // cube local variable n and return result

} // end function cubeByValue

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
