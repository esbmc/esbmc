// Fig. 21.22: fig21_22.cpp
// Standard library class map test program.
#include <iostream>
#include <iterator>
using std::cout;
using std::endl;

#include <map>  // map class-template definition

// define short name for map type used in this program
typedef std::map< int, double, std::less< int > > mid;

int main()
{
   mid pairs;

   // insert eight value_type objects in pairs
   pairs.insert( mid::value_type( 15, 2.7 ) );
   pairs.insert( mid::value_type( 30, 111.11 ) );
   pairs.insert( mid::value_type( 5, 1010.1 ) );
   pairs.insert( mid::value_type( 10, 22.22 ) );
   pairs.insert( mid::value_type( 25, 33.333 ) );
   pairs.insert( mid::value_type( 5, 77.54 ) ); // dupe ignored
   pairs.insert( mid::value_type( 20, 9.345 ) );
   pairs.insert( mid::value_type( 15, 99.3 ) ); // dupe ignored

   cout << "pairs contains:\nKey\tValue\n";
   
   // use const_iterator to walk through elements of pairs
   for ( mid::const_iterator iter = pairs.begin(); 
         iter != pairs.end(); ++iter )
      cout << iter->first << '\t' 
           << iter->second << '\n';

   // use subscript operator to change value for key 25
   pairs[ 25 ] = 9999.99;  
   
   // use subscript operator insert value for key 40
   pairs[ 40 ] = 8765.43;  
   
   cout << "\nAfter subscript operations, pairs contains:"
        << "\nKey\tValue\n";
   
   for ( mid::const_iterator iter2 = pairs.begin(); 
         iter2 != pairs.end(); ++iter2 )
      cout << iter2->first << '\t' 
           << iter2->second << '\n';

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
