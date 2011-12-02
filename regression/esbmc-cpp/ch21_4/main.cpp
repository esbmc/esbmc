// Fig. 21.17: fig21_17.cpp
// Standard library list class template test program.
#include <iostream>
#include <iterator>
using std::cout;
using std::endl;

#include <list>       // list class-template definition
#include <algorithm>  // copy algorithm

// prototype for function template printList
template < class T >
void printList( const std::list< T > &listRef );

int main()
{ 
   const int SIZE = 4;
   int array[ SIZE ] = { 2, 6, 4, 8 };

   std::list< int > values;
   std::list< int > otherValues;

   // insert items in values
   values.push_front( 1 );
   values.push_front( 2 );
   values.push_back( 4 );
   values.push_back( 3 );
   
   cout << "values contains: ";
   printList( values );

   values.sort();  // sort values

   cout << "\nvalues after sorting contains: ";
   printList( values );
  
   // insert elements of array into otherValues
   otherValues.insert( otherValues.begin(), 
      array, array + SIZE );

   cout << "\nAfter insert, otherValues contains: ";
   printList( otherValues );

   // remove otherValues elements and insert at end of values
   values.splice( values.end(), otherValues );
   
   cout << "\nAfter splice, values contains: ";
   printList( values );

   values.sort();  // sort values

   cout << "\nAfter sort, values contains: ";
   printList( values );

   // insert elements of array into otherValues
   otherValues.insert( otherValues.begin(), 
      array, array + SIZE );
   otherValues.sort();
   
   cout << "\nAfter insert, otherValues contains: ";
   printList( otherValues );
   
   // remove otherValues elements and insert into values 
   // in sorted order
   values.merge( otherValues );
   
   cout << "\nAfter merge:\n   values contains: ";
   printList( values );
   cout << "\n   otherValues contains: ";
   printList( otherValues );

   values.pop_front();  // remove element from front
   values.pop_back();   // remove element from back
   
   cout << "\nAfter pop_front and pop_back:" 
        << "\n   values contains: ";
   printList( values );
   
   values.unique();  // remove duplicate elements
   
   cout << "\nAfter unique, values contains: ";
   printList( values );

   // swap elements of values and otherValues
   values.swap( otherValues );
   
   cout << "\nAfter swap:\n   values contains: ";
   printList( values );
   cout << "\n   otherValues contains: ";
   printList( otherValues );

   // replace contents of values with elements of otherValues
   values.assign( otherValues.begin(), otherValues.end() );
   
   cout << "\nAfter assign, values contains: ";
   printList( values );

   // remove otherValues elements and insert into values 
   // in sorted order
   values.merge( otherValues ); 
   
   cout << "\nAfter merge, values contains: ";
   printList( values ); 
   
   values.remove( 4 );  // remove all 4s
   
   cout << "\nAfter remove( 4 ), values contains: ";
   printList( values );

   cout << endl;

   return 0;

} // end main

// printList function template definition; uses 
// ostream_iterator and copy algorithm to output list elements
template < class T >
void printList( const std::list< T > &listRef )
{
   if ( listRef.empty() )
      cout << "List is empty";

   else {
      std::ostream_iterator< T > output( cout, " " );
      std::copy( listRef.begin(), listRef.end(), output );

   } // end else

} // end function printList

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
