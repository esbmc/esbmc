/*******************************************************************\

Module: printf Formatting

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PRINTF_FORMATTER
#define CPROVER_PRINTF_FORMATTER

#include <util/irep2_expr.h>

class printf_formattert
{
public:
  void
  operator()(const std::string &format, const std::list<expr2tc> &_operands);

  void print(std::ostream &out);
  std::string as_string();

protected:
  std::string format;
  std::list<expr2tc> operands;
  std::list<expr2tc>::const_iterator next_operand;
  unsigned format_pos;
  inline bool eol() const
  {
    return format_pos >= format.size();
  }

  class eol_exception
  {
  };

  char next()
  {
    if(eol())
      throw eol_exception();
    return format[format_pos++];
  }

  void process_char(std::ostream &out);
  void process_format(std::ostream &out);

  const expr2tc make_type(const expr2tc &src, const type2tc &dest);
};

#endif
