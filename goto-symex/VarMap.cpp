// Bruno J. Savino
#include <iostream>

#include <cstring>
#include "VarMap.h"


// default VarMap constructor
VarMap::VarMap( std::string varName,
   std::string varInfo )
{
   setVarName( varName );
   setVarInfo( varInfo );


} // end VarMap constructor


std::string VarMap::getVarName() const
{
   return varName;

} 

void VarMap::setVarName( std::string varNameString )
{

   const char *varNameValue = varNameString.data();
   int length = strlen( varNameValue );
   length = ( length < 50 ? length : 49 );
   strncpy( varName, varNameValue, length );


   varName[ length ] = '\0';

} 

std::string VarMap::getVarInfo() const
{
   return varInfo;

} 

void VarMap::setVarInfo( std::string varInfoString )
{
   
   const char *varInfoValue = varInfoString.data();
   int length = strlen( varInfoValue );
   length = ( length < 30 ? length : 29 );
   strncpy( varInfo, varInfoValue, length );


   varInfo[ length ] = '\0';
} 
