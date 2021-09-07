/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

/// \file
/// C++ Language Type Checking

#include "padding.h"

#include <algorithm>

#include <util/arith_tools.h>
#include <util/config.h>
#include <util/simplify_expr.h>
#include <util/type_byte_size.h>

BigInt alignment(const typet &type, const namespacet &ns)
{
  // we need to consider a number of different cases:
  // - alignment specified in the source, which will be recorded in
  // "alignment"
  // - alignment induced by packing ("The alignment of a member will
  // be on a boundary that is either a multiple of n or a multiple of
  // the size of the member, whichever is smaller."); both
  // "alignment" and "packed" will be set
  // - natural alignment, when neither "alignment" nor "packed"
  // are set
  // - dense packing with only "packed" set.

  // is the alignment given?
  const exprt &given_alignment =
    static_cast<const exprt &>(type.find("alignment"));

  BigInt a_int = 0;

  // we trust it blindly, no matter how nonsensical
  if(given_alignment.is_not_nil())
    a_int = string2integer(given_alignment.cformat().as_string());

  // alignment but no packing
  if(a_int > 0 && !type.get_bool("packed"))
    return a_int;
  // no alignment, packing
  else if(a_int == 0 && type.get_bool("packed"))
    return 1;

  // compute default
  BigInt result = 0;

  if(type.id() == typet::t_array)
    result = alignment(type.subtype(), ns);
  else if(type.id() == typet::t_struct || type.id() == typet::t_union)
  {
    result = 1;

    // get the max
    // (should really be the smallest common denominator)
    for(const auto &c : to_struct_union_type(type).components())
      result = std::max(result, alignment(c.type(), ns));
  }
  else if(type.get_bool("#bitfield"))
  {
    // we align these according to the 'underlying type'
    result = alignment(type.subtype(), ns);
  }
  else if(
    type.id() == typet::t_unsignedbv || type.id() == typet::t_signedbv ||
    type.id() == typet::t_fixedbv || type.id() == typet::t_floatbv ||
    type.id() == typet::t_bool || type.id() == typet::t_pointer)
  {
    type2tc thetype = migrate_type(type);
    result = type_byte_size(thetype);
  }
  else if(type.id() == typet::t_symbol)
    result = alignment(ns.follow(type), ns);
  else
    result = 1;

  // if an alignment had been provided and packing was requested, take
  // the smallest alignment
  if(a_int > 0 && a_int < result)
    result = a_int;

  return result;
}

static struct_typet::componentst::iterator pad_bit_field(
  struct_typet::componentst &components,
  struct_typet::componentst::iterator where,
  std::size_t pad_bits)
{
  const unsignedbv_typet padding_type(pad_bits);

  std::string index = std::to_string(where - components.begin());
  struct_typet::componentt component(
    "bit_field_pad$" + index, "anon_bit_field_pad$" + index, padding_type);

  component.type().set("#bitfield", true);
  component.set_is_padding(true);
  return std::next(components.insert(where, component));
}

static struct_typet::componentst::iterator pad(
  struct_typet::componentst &components,
  struct_typet::componentst::iterator where,
  std::size_t pad_bits)
{
  const unsignedbv_typet padding_type(pad_bits);

  std::string index = std::to_string(where - components.begin());
  struct_typet::componentt component(
    "pad$" + index, "anon_pad$" + index, padding_type);

  component.set_is_padding(true);
  return std::next(components.insert(where, component));
}

void add_padding(struct_typet &type, const namespacet &ns)
{
  struct_typet::componentst &components = type.components();

  // First make bit-fields appear on byte boundaries
  {
    std::size_t bit_field_bits = 0;

    for(struct_typet::componentst::iterator it = components.begin();
        it != components.end();
        it++)
    {
      if(
        it->type().get_bool("#bitfield") &&
        string2integer(it->type().width().as_string()) != 0)
      {
        // count the bits
        bit_field_bits +=
          string2integer(it->type().width().as_string()).to_uint64();
      }
      else if(bit_field_bits != 0)
      {
        // not on a byte-boundary?
        if((bit_field_bits % config.ansi_c.char_width) != 0)
        {
          const std::size_t pad = config.ansi_c.char_width -
                                  bit_field_bits % config.ansi_c.char_width;
          it = pad_bit_field(components, it, pad);
        }

        bit_field_bits = 0;
      }
    }

    // Add padding at the end?
    if((bit_field_bits % config.ansi_c.char_width) != 0)
    {
      const std::size_t pad =
        config.ansi_c.char_width - bit_field_bits % config.ansi_c.char_width;
      pad_bit_field(components, components.end(), pad);
    }
  }

  // Is the struct packed, without any alignment specification?
  if(type.get_bool("packed") && type.find("alignment").is_nil())
    return; // done

  BigInt offset = 0;
  BigInt max_alignment = 0;
  std::size_t bit_field_bits = 0;

  for(struct_typet::componentst::iterator it = components.begin();
      it != components.end();
      it++)
  {
    const typet it_type = it->type();
    BigInt a = 1;

    const bool packed =
      it_type.get_bool("packed") || ns.follow(it_type).get_bool("packed");

    if(it_type.get_bool("#bitfield"))
    {
      a = alignment(it_type, ns);

      // A zero-width bit-field causes alignment to the base-type.
      if(string2integer(it_type.width().as_string()) == 0)
      {
      }
      else
      {
        // Otherwise, ANSI-C says that bit-fields do not get padded!
        // We consider the type for max_alignment, however.
        if(max_alignment < a)
          max_alignment = a;

        std::size_t w = string2integer(it_type.width().as_string()).to_uint64();
        bit_field_bits += w;
        const std::size_t bytes = bit_field_bits / config.ansi_c.char_width;
        bit_field_bits %= config.ansi_c.char_width;
        offset += bytes;
        continue;
      }
    }
    else
      a = alignment(it_type, ns);

    assert(bit_field_bits == 0);

    // check minimum alignment
    if(a < config.ansi_c.alignment && !packed)
      a = config.ansi_c.alignment;

    if(max_alignment < a)
      max_alignment = a;

    if(a != 1)
    {
      // we may need to align it
      const BigInt displacement = offset % a;

      if(displacement != 0)
      {
        const BigInt pad_bytes = a - displacement;
        const std::size_t pad_bits =
          (pad_bytes * config.ansi_c.char_width).to_uint64();
        it = pad(components, it, pad_bits);
        offset += pad_bytes;
      }
    }

    type2tc thetype = migrate_type(it_type);
    offset += type_byte_size(thetype);
  }

  // any explicit alignment for the struct?
  const exprt &alignment = static_cast<const exprt &>(type.find("alignment"));
  if(alignment.is_not_nil())
  {
    const auto tmp_i = string2integer(alignment.cformat().as_string());
    if(tmp_i > max_alignment)
      max_alignment = tmp_i;
  }
  // Is the struct packed, without any alignment specification?
  else if(type.get_bool("packed"))
    return; // done

  // There may be a need for 'end of struct' padding.
  // We use 'max_alignment'.
  if(max_alignment > 1)
  {
    // we may need to align it
    BigInt displacement = offset % max_alignment;
    if(displacement != 0)
    {
      BigInt pad_bytes = max_alignment - displacement;
      std::size_t pad_bits = (pad_bytes * config.ansi_c.char_width).to_uint64();
      pad(components, components.end(), pad_bits);
    }
  }
}

void add_padding(union_typet &type, const namespacet &ns)
{
  BigInt max_alignment_bits = alignment(type, ns) * config.ansi_c.char_width;
  BigInt size_bits = 0;

  // check per component, and ignore those without fixed size
  for(const auto &c : type.components())
  {
    type2tc thetype = migrate_type(c.type());
    size_bits = std::max(size_bits, type_byte_size(thetype));
  }

  // Is the union packed?
  if(type.get_bool("packed"))
  {
    // The size needs to be a multiple of 1 char only.
    max_alignment_bits = config.ansi_c.char_width;
  }

  // The size must be a multiple of the alignment, or
  // we add a padding member to the union.

  if(size_bits % max_alignment_bits != 0)
  {
    BigInt padding_bits = max_alignment_bits - (size_bits % max_alignment_bits);
    unsignedbv_typet padding_type((size_bits + padding_bits).to_uint64());

    struct_typet::componentt component;
    component.type() = padding_type;
    component.set_name("$pad");
    component.set_is_padding(true);

    type.components().push_back(component);
  }
}
