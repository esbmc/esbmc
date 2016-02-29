/*******************************************************************\

Module: ANSI-C Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_CONVERT_TYPE_H
#define CPROVER_ANSI_C_CONVERT_TYPE_H

#include <message_stream.h>
#include <c_types.h>

#include <ansi-c/c_qualifiers.h>
#include <ansi-c/c_storage_spec.h>

class ansi_c_convert_typet:public message_streamt
{
public:
  unsigned unsigned_cnt, signed_cnt, char_cnt,
           int_cnt, short_cnt, long_cnt,
           double_cnt, float_cnt, bool_cnt,
           int8_cnt, int16_cnt, int32_cnt, int64_cnt,
           ptr32_cnt, ptr64_cnt;

  // storage spec
  c_storage_spect c_storage_spec;

  // qualifiers
  c_qualifierst c_qualifiers;

  void read(const typet &type);
  void write(typet &type);

  locationt location;

  std::list<typet> other;

  ansi_c_convert_typet(message_handlert &_message_handler):
    message_streamt(_message_handler)
  {
  }

  void clear()
  {
    unsigned_cnt=signed_cnt=char_cnt=int_cnt=short_cnt=
    long_cnt=double_cnt=float_cnt=bool_cnt=
    int8_cnt=int16_cnt=int32_cnt=int64_cnt=
    ptr32_cnt=ptr64_cnt=0;

    other.clear();
    c_storage_spec.clear();
    c_qualifiers.clear();
  }

protected:
  void read_rec(const typet &type);
};

#endif
