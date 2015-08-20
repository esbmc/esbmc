// Fig. 21.32: fig21_32.cpp
// Standard library algorithms iter_swap, swap and swap_ranges.
#include <iostream>
#include <string>
#include <iterator> //Foi adicionada a biblioteca "iterator".
using std::cout;
using std::endl;

#include <algorithm>  // algorithm definitions

int main()
{
   const int SIZE = 10;
   int a[ SIZE ] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   std::ostream_iterator< int > output( cout, " " );

   cout << "Array a contains:\n   ";
   std::copy( a, a + SIZE, output );

   // swap elements at locations 0 and 1 of array a
   std::swap( a[ 0 ], a[ 1 ] );

   cout << "\nArray a after swapping a[0] and a[1] "
        << "using swap:\n   ";
   std::copy( a, a + SIZE, output );

   // use iterators to swap elements at locations 
   // 0 and 1 of array a
   std::iter_swap( &a[ 0 ], &a[ 1 ] );
   cout << "\nArray a after swapping a[0] and a[1] "
        << "using iter_swap:\n   ";
   std::copy( a, a + SIZE, output );

   // swap elements in first five elements of array a with
   // elements in last five elements of array a 
   std::swap_ranges( a, a + 5, a + 5 );

   cout << "\nArray a after swapping the first five elements\n"
        << "with the last five elements:\n   ";
   std::copy( a, a + SIZE, output );

   cout << endl;

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
