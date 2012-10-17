// Fig. 17.19: fig17_19.cpp
// Tree class test program.
#include <iostream>

using std::cout;
using std::cin;
using std::fixed;

#include <iomanip>
using std::setprecision;

#include "tree.h"  // Tree class definition

int main()
{
   Tree< int > intTree;  // create Tree of int values
   int intValue;

   cout << "Enter 10 integer values:\n";

   for( int i = 0; i < 10; i++ ) {
      cin >> intValue;
      intTree.insertNode( intValue );

   } // end for

   cout << "\nPreorder traversal\n";
   intTree.preOrderTraversal();

   cout << "\nInorder traversal\n";
   intTree.inOrderTraversal();

   cout << "\nPostorder traversal\n";
   intTree.postOrderTraversal();

   Tree< double > doubleTree;  // create Tree of double values
   double doubleValue;

   cout << fixed << setprecision( 1 )
        << "\n\n\nEnter 10 double values:\n";

   for ( int j = 0; j < 10; j++ ) {
      cin >> doubleValue;
      doubleTree.insertNode( doubleValue );

   } // end for

   cout << "\nPreorder traversal\n";
   doubleTree.preOrderTraversal();

   cout << "\nInorder traversal\n";
   doubleTree.inOrderTraversal();

   cout << "\nPostorder traversal\n";
   doubleTree.postOrderTraversal();

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