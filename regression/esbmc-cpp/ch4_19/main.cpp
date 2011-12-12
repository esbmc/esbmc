// Fig. 4.20: fig04_20.cpp
// Binary search of an array.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include <iomanip>

using std::setw;

// function prototypes
int binarySearch( const int [], int, int, int, int );
void printHeader( int );
void printRow( const int [], int, int, int, int );

int main()
{
   const int arraySize = 15;  // size of array a
   int a[ arraySize ];        // create array a
   int key;                   // value to locate in a

   for ( int i = 0; i < arraySize; i++ )  // create some data
      a[ i ] = 2 * i;   

   cout << "Enter a number between 0 and 28: ";
   cin >> key;

   printHeader( arraySize );

   // search for key in array a
   int result = 
      binarySearch( a, key, 0, arraySize - 1, arraySize );

   // display results
   if ( result != -1 )
      cout << '\n' << key << " found in array element "
           << result << endl;
   else
      cout << '\n' << key << " not found" << endl;

   return 0;  // indicates successful termination

} // end main

// function to perform binary search of an array
int binarySearch( const int b[], int searchKey, int low, 
   int high, int size )
{
   int middle;

   // loop until low subscript is greater than high subscript
   while ( low <= high ) {

      // determine middle element of subarray being searched
      middle = ( low + high ) / 2;  

      // display subarray used in this loop iteration
      printRow( b, low, middle, high, size );

      // if searchKey matches middle element, return middle
      if ( searchKey == b[ middle ] )  // match
         return middle;

      else 

         // if searchKey less than middle element, 
         // set new high element
         if ( searchKey < b[ middle ] )
            high = middle - 1;  // search low end of array

         // if searchKey greater than middle element, 
         // set new low element
         else
            low = middle + 1;   // search high end of array
   }

   return -1;  // searchKey not found

} // end function binarySearch

// print header for output
void printHeader( int size )
{
   cout << "\nSubscripts:\n";

   // output column heads
   for ( int j = 0; j < size; j++ )
      cout << setw( 3 ) << j << ' ';

   cout << '\n';  // start new line of output

   // output line of - characters
   for ( int k = 1; k <= 4 * size; k++ )
      cout << '-';

   cout << endl;  // start new line of output

} // end function printHeader

// print one row of output showing the current
// part of the array being processed
void printRow( const int b[], int low, int mid, 
   int high, int size )
{
   // loop through entire array
   for ( int m = 0; m < size; m++ )

      // display spaces if outside current subarray range
      if ( m < low || m > high )
         cout << "    ";

      // display middle element marked with a *
      else 

         if ( m == mid )           // mark middle value
            cout << setw( 3 ) << b[ m ] << '*';  

         // display other elements in subarray
         else
            cout << setw( 3 ) << b[ m ] << ' ';

   cout << endl;  // start new line of output

} // end function printRow


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
