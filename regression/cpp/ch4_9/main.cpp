// Fig. 4.9: fig04_09.cpp
// Histogram printing program.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

int main()
{
   const int arraySize = 10;
   int n[ arraySize ] = { 19, 3, 15, 7, 11, 9, 13, 5, 17, 1 };

   cout << "Element" << setw( 13 ) << "Value"
        << setw( 17 ) << "Histogram" << endl;

   // for each element of array n, output a bar in histogram
   for ( int i = 0; i < arraySize; i++ ) {
      cout << setw( 7 ) << i << setw( 13 ) 
           << n[ i ] << setw( 9 );

      for ( int j = 0; j < n[ i ]; j++ )   // print one bar
         cout << '*';

      cout << endl;  // start next line of output

   } // end outer for structure

   return 0;  // indicates successful termination

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
