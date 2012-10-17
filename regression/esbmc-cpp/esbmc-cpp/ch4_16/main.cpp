// Fig. 4.16: fig04_16.cpp
// This program sorts an array's values into ascending order.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setw;

int main()
{
   const int arraySize = 10;  // size of array a
   int a[ arraySize ] = { 2, 6, 4, 8, 10, 12, 89, 68, 45, 37 };
   int hold;  // temporary location used to swap array elements

   cout << "Data items in original order\n";

   // output original array
   for ( int i = 0; i < arraySize; i++ )
      cout << setw( 4 ) << a[ i ];

   // bubble sort
   // loop to control number of passes
   for ( int pass = 0; pass < arraySize - 1; pass++ )

      // loop to control number of comparisons per pass
      for ( int j = 0; j < arraySize - 1; j++ )  

         // compare side-by-side elements and swap them if 
         // first element is greater than second element
         if ( a[ j ] > a[ j + 1 ] ) {      
            hold = a[ j ];        
            a[ j ] = a[ j + 1 ]; 
            a[ j + 1 ] = hold;  
            
         } // end if

   cout << "\nData items in ascending order\n";

   // output sorted array
   for ( int k = 0; k < arraySize; k++ )
      cout << setw( 4 ) << a[ k ];

   cout << endl;

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
