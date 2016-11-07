/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_FORMAT_SPEC_H
#define CPROVER_FORMAT_SPEC_H

class format_spect
{
public:
  unsigned min_width;
  unsigned precision;
  bool zero_padding;
  typedef enum { DECIMAL, SCIENTIFIC, AUTOMATIC } stylet;
  stylet style;
  
  format_spect():
    min_width(0),
    precision(6),
    zero_padding(false),
    style(AUTOMATIC)
  {
  }

  explicit format_spect(stylet _style):
    min_width(0),
    precision(6),
    zero_padding(false),
    style(_style)
  {
  }

  static format_spect scientific()
  {
    return format_spect(SCIENTIFIC);
  }

  static format_spect automatic()
  {
    return format_spect(AUTOMATIC);
  }
};

#endif
