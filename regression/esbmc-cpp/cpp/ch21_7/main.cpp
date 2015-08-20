// Fig. 21.20: fig21_20.cpp
// Standard library class set test program.
#include <iostream>
#include <iterator>
using std::cout;
using std::endl;

#include <set>

// define short name for set type used in this program
typedef std::set< double, std::less< double > > double_set;

#include <algorithm>


#include <ostream>




int main()
{

   const int SIZE = 5;
   double a[ SIZE ] = { 2.1, 4.2, 9.5, 2.1, 3.7 };   

   double_set doubleSet( a, a + SIZE );

  std::ostream_iterator< double > output( cout, " " );

   cout << "doubleSet contains: ";
   std::copy( doubleSet.begin(), doubleSet.end(), output );

   // p represents pair containing const_iterator and bool
   std::pair< double_set::const_iterator, bool > p;

   // insert 13.8 in doubleSet; insert returns pair in which 
   // p.first represents location of 13.8 in doubleSet and 
   // p.second represents whether 13.8 was inserted
   p = doubleSet.insert( 13.8 ); // value not in set

   cout << "\n\n" << *( p.first ) 
        << ( p.second ? " was" : " was not" ) << " inserted";

   cout << "\ndoubleSet contains: ";
   std::copy( doubleSet.begin(), doubleSet.end(), output );

   // insert 9.5 in doubleSet
   p = doubleSet.insert( 9.5 );  // value already in set

   cout << "\n\n" << *( p.first ) 
        << ( p.second ? " was" : " was not" ) << " inserted";
   
   cout << "\ndoubleSet contains: ";
   std::copy( doubleSet.begin(), doubleSet.end(), output );

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
