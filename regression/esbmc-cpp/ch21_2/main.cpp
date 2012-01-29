// Fig. 21.14: fig21_14.cpp
// Demonstrating standard library vector class template.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include <vector>  // vector class-template definition

// prototype for function template printVector
template < class T >
void printVector( const std::vector< T > &integers2 );

int main()
{
   const int SIZE = 6;   
   int array[ SIZE ] = { 1, 2, 3, 4, 5, 6 };

   std::vector< int > integers;

   cout << "The initial size of integers is: " 
        << integers.size()
        << "\nThe initial capacity of integers is: " 
        << integers.capacity();

   // function push_back is in every sequence collection
   integers.push_back( 2 );  
   integers.push_back( 3 );  
   integers.push_back( 4 );

   cout << "\nThe size of integers is: " << integers.size()
        << "\nThe capacity of integers is: " 
        << integers.capacity();

   cout << "\n\nOutput array using pointer notation: ";

   for ( int *ptr = array; ptr != array + SIZE; ++ptr )
      cout << *ptr << ' ';

   cout << "\nOutput vector using iterator notation: ";
   printVector( integers );

   cout << "\nReversed contents of vector integers: ";

   std::vector< int >::reverse_iterator reverseIterator;

   for ( reverseIterator = integers.rbegin(); 
         reverseIterator!= integers.rend(); 
         ++reverseIterator )
      cout << *reverseIterator << ' ';

   cout << endl;

   return 0;

} // end main

// function template for outputting vector elements
template < class T >
void printVector( const std::vector< T > &integers2 )
{
   std::vector< T >::const_iterator constIterator;

   for ( constIterator = integers2.begin(); 
         constIterator != integers2.end(); 
         constIterator++ )
      cout << *constIterator << ' ';

} // end function printVector

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
