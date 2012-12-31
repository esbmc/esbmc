// Fig. 16.23: viewcart.cpp
// Program to view books in the shopping cart.
#include <iostream>

using std::cerr;
using std::cout;
using std::cin;
using std::ios;

#include <istream>

#include <fstream>

using std::ifstream;
using std::ofstream;

#include <string>

using std::string;

#include <cstdlib>

void outputBooks( const string &, const string & );

int main()
{
   // variable to store query string
   char query[ 1024 ] = ""; 
   char *cartData; // variable to hold contents of cart

   string dataString = "";
   string cookieString = "";
   string isbnEntered = "";
   int contentLength = 0;

   // retrieve cookie data
   if ( getenv( "HTTP_COOKIE" ) ) {
      cartData = getenv( "HTTP_COOKIE" );
      cookieString = cartData;
   } // end if
   
   // data was entered
   if ( getenv( "CONTENT_LENGTH" ) ) {
      contentLength = atoi( getenv( "CONTENT_LENGTH" ) );
      cin.read( query, contentLength );
      dataString = query;
      
      // find location of isbn value
      int addLocation = dataString.find( "add=" ) + 4;
      int endAdd = dataString.find( "&isbn" );
      int isbnLocation = dataString.find( "isbn=" ) + 5;

      // retrieve isbn number to add to cart
      isbnEntered = dataString.substr( isbnLocation );
      
      // write cookie
      string expires = "Friday, 14-MAY-10 16:00:00 GMT";
      int cartLocation = cookieString.find( "CART=" ) + 5;
      
      // cookie exists
      if ( cartLocation > 0 )
         cookieString = cookieString.substr( cartLocation );

      // no cookie data exists
      if ( cookieString == "" )
         cookieString = isbnEntered;

      // cookie data exists
      else
         cookieString = cookieString + "," + isbnEntered;
      
      // set cookie
      cout << "set-cookie: CART=" << cookieString << "; expires="
           << expires << "; path=\n";

   } // end if
   
   // output header
   cout << "Content-type: text/html\n\n";
   
   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>"
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\" \"http://www.w3.org/TR/xhtml1"
        << "/DTD/xhtml1-transitional.dtd\">";
   
   // output html element and some of its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Shopping Cart</title></head>"
        << "<body><center>"
        << "<p>Here is your current order:</p>";
 
   // cookie data exists
   if ( cookieString != "" )
      outputBooks( cookieString, isbnEntered );
      
   cout << "</body></html>\n";
   return 0;

} // end main

// function to output books in catalog.txt
void outputBooks( const string &cookieRef, const string &isbnRef )
{
   char book[ 50 ] = "";
   char year[ 50 ] = "";
   char isbn[ 50 ] = "";
   char price[ 50 ] = "";

   string bookString = "";
   string yearString = "";
   string isbnString = "";
   string priceString = "";

   // open file for input
   ifstream userData( "catalog.txt", ios::in );

   // file could not be opened
   if ( !userData ) {
      cerr << "Could not open database.";
      exit( 1 );
   } // end if

   // output link to log out and table to display books
   cout << "<a href=\"/cgi-bin/logout.cgi\">Sign Out";
   cout << "</a><br><br>";
   cout << "<table border = 1 cellpadding = 7 >";

   // file is open
   while ( userData ) {

      // retrieve book information
      userData.getline( book, 50 );
      bookString = book;

      // retrieve year information
      userData.getline( year, 50 );
      yearString = year;

      // retrieve isbn number
      userData.getline( isbn, 50 );
      isbnString = isbn;

      // retrieve price
      userData.getline( price, 50 );
      priceString = price;
           
      int match = cookieRef.find( isbn );

      // match has been made
      if ( match > 0 || isbnRef == isbnString ) {

         // output table row with book information
         cout << "<tr>"
              << "<form method=\"post\""
              << "action=\"/cgi-bin/viewcart.cgi\">"
              << "<td>" << bookString << "</td>"
              << "<td>" << yearString << "</td>"
              << "<td>" << isbnString << "</td>"
              << "<td>" << priceString << "</td>";

      } // end if

      cout << "</form></tr>";

   } // end while

   // output link to add more books
   cout << "<a href=\"/cgi-bin/shop.cgi\">Back to book list</a>";

} // end outputBooks

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