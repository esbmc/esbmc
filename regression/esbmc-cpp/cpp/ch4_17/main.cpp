// Fig. 4.17: fig04_17.cpp
// This program introduces the topic of survey data analysis.
// It computes the mean, median, and mode of the data.
#include <iostream>

using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>

using std::setw;
using std::setprecision;

void mean( const int [], int );
void median( int [], int );
void mode( int [], int [], int );
void bubbleSort( int[], int );
void printArray( const int[], int );

int main()
{
   const int responseSize = 99;  // size of array responses

   int frequency[ 10 ] = { 0 };  // initialize array frequency

   // initialize array responses
   int response[ responseSize ] =  
          { 6, 7, 8, 9, 8, 7, 8, 9, 8, 9,
            7, 8, 9, 5, 9, 8, 7, 8, 7, 8,
            6, 7, 8, 9, 3, 9, 8, 7, 8, 7,
            7, 8, 9, 8, 9, 8, 9, 7, 8, 9,
            6, 7, 8, 7, 8, 7, 9, 8, 9, 2,
            7, 8, 9, 8, 9, 8, 9, 7, 5, 3,
            5, 6, 7, 2, 5, 3, 9, 4, 6, 4,
            7, 8, 9, 6, 8, 7, 8, 9, 7, 8,
            7, 4, 4, 2, 5, 3, 8, 7, 5, 6,
            4, 5, 6, 1, 6, 5, 7, 8, 7 };

   // process responses
   mean( response, responseSize );
   median( response, responseSize );
   mode( frequency, response, responseSize );

   return 0;  // indicates successful termination

} // end main

// calculate average of all response values
void mean( const int answer[], int arraySize )
{
   int total = 0;

   cout << "********\n  Mean\n********\n";

   // total response values
   for ( int i = 0; i < arraySize; i++ )
      total += answer[ i ];

   // format and output results
   cout << fixed << setprecision( 4 );

   cout << "The mean is the average value of the data\n"
        << "items. The mean is equal to the total of\n"
        << "all the data items divided by the number\n"
        << "of data items (" << arraySize 
        << "). The mean value for\nthis run is: " 
        << total << " / " << arraySize << " = "
        << static_cast< double >( total ) / arraySize 
        << "\n\n";

} // end function mean

// sort array and determine median element's value
void median( int answer[], int size )
{
   cout << "\n********\n Median\n********\n"
        << "The unsorted array of responses is";

   printArray( answer, size );  // output unsorted array

   bubbleSort( answer, size );  // sort array

   cout << "\n\nThe sorted array is";
   printArray( answer, size );  // output sorted array 
    
   // display median element
   cout << "\n\nThe median is element " << size / 2
        << " of\nthe sorted " << size 
        << " element array.\nFor this run the median is "
        << answer[ size / 2 ] << "\n\n";

} // end function median

// determine most frequent response
void mode( int freq[], int answer[], int size )
{
   int largest = 0;    // represents largest frequency
   int modeValue = 0;  // represents most frequent response

   cout << "\n********\n  Mode\n********\n";

   // initialize frequencies to 0
   for ( int i = 1; i <= 9; i++ )
      freq[ i ] = 0;

   // summarize frequencies
   for ( int j = 0; j < size; j++ )
      ++freq[ answer[ j ] ];

   // output headers for result columns
   cout << "Response" << setw( 11 ) << "Frequency"
        << setw( 19 ) << "Histogram\n\n" << setw( 55 )
        << "1    1    2    2\n" << setw( 56 )
        << "5    0    5    0    5\n\n";

   // output results
   for ( int rating = 1; rating <= 9; rating++ ) {
      cout << setw( 8 ) << rating << setw( 11 )
           << freq[ rating ] << "          ";

      // keep track of mode value and largest fequency value
      if ( freq[ rating ] > largest ) {
         largest = freq[ rating ];
         modeValue = rating;

      } // end if

      // output histogram bar representing frequency value
      for ( int k = 1; k <= freq[ rating ]; k++ )
         cout << '*';

      cout << '\n';  // begin new line of output

   } // end outer for

   // display the mode value
   cout << "The mode is the most frequent value.\n"
        << "For this run the mode is " << modeValue
        << " which occurred " << largest << " times." << endl;

} // end function mode

// function that sorts an array with bubble sort algorithm
void bubbleSort( int a[], int size )
{
   int hold;  // temporary location used to swap elements

   // loop to control number of passes
   for ( int pass = 1; pass < size; pass++ )

      // loop to control number of comparisons per pass
      for ( int j = 0; j < size - 1; j++ )

         // swap elements if out of order
         if ( a[ j ] > a[ j + 1 ] ) {
            hold = a[ j ];
            a[ j ] = a[ j + 1 ];
            a[ j + 1 ] = hold;

         } // end if

} // end function bubbleSort

// output array contents (20 values per row)
void printArray( const int a[], int size )
{
   for ( int i = 0; i < size; i++ ) {

      if ( i % 20 == 0 )  // begin new line every 20 values
         cout << endl;

      cout << setw( 2 ) << a[ i ];

   } // end for

} // end function printArray


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
