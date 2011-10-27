// Fig. 3.8: fig03_08.cpp
// Roll a six-sided die 6000 times.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

#include <cstdlib>   // contains function prototype for rand

int main()
{
   int frequency1 = 0;
   int frequency2 = 0;
   int frequency3 = 0; 
   int frequency4 = 0;
   int frequency5 = 0; 
   int frequency6 = 0;
   int face;  // represents one roll of the die

   // loop 6000 times and summarize results
   for ( int roll = 1; roll <= 6000; roll++ ) {
      face = 1 + rand() % 6;  // random number from 1 to 6

      // determine face value and increment appropriate counter
      switch ( face ) {

         case 1:          // rolled 1
            ++frequency1;
            break;
         
         case 2:          // rolled 2
            ++frequency2;
            break;
         
         case 3:          // rolled 3
            ++frequency3;
            break;
         
         case 4:          // rolled 4
            ++frequency4;
            break;
         
         case 5:          // rolled 5
            ++frequency5;
            break;
         
         case 6:          // rolled 6
            ++frequency6;
            break;
         
         default:         // invalid value
            cout << "Program should never get here!";

      } // end switch 

   } // end for 

   // display results in tabular format
   cout << "Face" << setw( 13 ) << "Frequency"
        << "\n   1" << setw( 13 ) << frequency1
        << "\n   2" << setw( 13 ) << frequency2
        << "\n   3" << setw( 13 ) << frequency3
        << "\n   4" << setw( 13 ) << frequency4
        << "\n   5" << setw( 13 ) << frequency5
        << "\n   6" << setw( 13 ) << frequency6 << endl;

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
