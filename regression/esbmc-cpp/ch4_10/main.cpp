// Fig. 4.10: fig04_10.cpp
// Roll a six-sided die 6000 times.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

#include <cstdlib>
#include <ctime>

int main()
{
   const int arraySize = 7;
   int frequency[ arraySize ] = { 0 };

   srand( time( 0 ) );  // seed random-number generator

   // roll die 6000 times
   for ( int roll = 1; roll <= 6000; roll++ )       
      ++frequency[ 1 + rand() % 6 ]; // replaces 20-line switch
                                     // of Fig. 3.8

   cout << "Face" << setw( 13 ) << "Frequency" << endl;

   // output frequency elements 1-6 in tabular format
   for ( int face = 1; face < arraySize; face++ )  
      cout << setw( 4 ) << face
           << setw( 13 ) << frequency[ face ] << endl;

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
