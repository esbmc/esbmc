// Fig. 4.19: fig04_19.cpp
// Linear search of an array.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

int linearSearch( const int [], int, int );  // prototype

int main()
{
   const int arraySize = 100;  // size of array a
   int a[ arraySize ];         // create array a
   int searchKey;              // value to locate in a

   for ( int i = 0; i < arraySize; i++ )  // create some data
      a[ i ] = 2 * i;

   cout << "Enter integer search key: ";
   cin >> searchKey;

   // attempt to locate searchKey in array a
   int element = linearSearch( a, searchKey, arraySize );

   // display results
   if ( element != -1 )
      cout << "Found value in element " << element << endl;
   else
      cout << "Value not found" << endl;

   return 0;  // indicates successful termination

} // end main

// compare key to every element of array until location is 
// found or until end of array is reached; return subscript of 
// element if key or -1 if key not found
int linearSearch( const int array[], int key, int sizeOfArray )
{
   for ( int j = 0; j < sizeOfArray; j++ )

      if ( array[ j ] == key )  // if found,
         return j;              // return location of key

   return -1;  // key not found

} // end function linearSearch

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
