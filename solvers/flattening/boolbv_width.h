/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BOOLBV_WIDTH_H
#define CPROVER_BOOLBV_WIDTH_H

#include <type.h>

#define BV_ADDR_BITS 8

class boolbv_widtht
{
 public:
  boolbv_widtht();
  virtual ~boolbv_widtht();
 
  virtual bool get_width(const typet &type, unsigned &width) const;
};

bool boolbv_get_width(const typet &type, unsigned &width);

#endif
