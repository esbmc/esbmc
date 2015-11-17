/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_LOCATION_H
#define CPROVER_LOCATION_H

#include "irep.h"

class locationt:public irept
{
public:
  std::string as_string() const;
  
  const irep_idt &get_file() const
  {
    return file();
  }

  const irep_idt &get_line() const
  {
    return line();
  }

  const irep_idt &get_column() const
  {
    return column();
  }

  const irep_idt &get_function() const
  {
    return function();
  }

  void set_file(const irep_idt &file)
  {
    this->file(file);
  }

  void set_line(const irep_idt &line)
  {
    this->line(line);
  }

  void set_line(unsigned line)
  {
    this->line(line);
  }

  void set_column(const irep_idt &column)
  {
    set("column", column);
  }

  void set_column(unsigned column)
  {
    set("column", column);
  }

  void set_function(const irep_idt &function)
  {
    this->function(function);
  }

};

std::ostream &operator <<(std::ostream &out, const locationt &location);

#endif
