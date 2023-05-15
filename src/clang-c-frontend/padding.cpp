/// \file
/// C++ Language Type Checking

#include "padding.h"

#include <algorithm>

#include <util/arith_tools.h>
#include <util/config.h>
#include <util/simplify_expr.h>
#include <util/type_byte_size.h>

static std::size_t ext_int_representation_bytes(const typet &type)
{
  // We represent an ExtInt with the smallest integer type that can hold it
  // TODO: This should be limited to a maximum of 8 bytes, but pointer analysis
  // currently expects the fields to be aligned to the least power of 2 greater
  // than the width in bytes
  const std::size_t bits = string2integer(type.width().as_string()).to_uint64();

  std::size_t result;
  for(result = 1; bits > result * config.ansi_c.char_width; result *= 2)
    ;

  return result;
}

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
  else if(type.get_bool("#extint"))
    result = ext_int_representation_bytes(type);
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

static struct_typet::componentst::iterator pad_ext_int_after(
  struct_typet::componentst &components,
  struct_typet::componentst::iterator where,
  std::size_t pad_bits)
{
  where = std::next(where);
  const unsignedbv_typet padding_type(pad_bits);

  std::string index = std::to_string(where - components.begin());
  struct_typet::componentt component(
    "ext_int_pad$" + index, "anon_ext_int_pad$" + index, padding_type);

  component.type().set("#extint", true);
  component.set_is_padding(true);
  return components.insert(where, component);
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
      bool is_bitfield = it->type().get_bool("#bitfield");
      bool is_extint = it->type().get_bool("#extint");
      irep_idt width = it->type().width();

      /* Bitfields and _ExtInt need their width set. */
      assert(!(is_bitfield || is_extint) || !width.empty());

      size_t w = string2integer(width.as_string()).to_uint64();

      if(is_bitfield && w != 0)
      {
        // count the bits
        bit_field_bits += w;
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

      // Pad out extints that arent in bitfields
      if(is_extint && !is_bitfield)
      {
        assert(bit_field_bits == 0);

        // Pad to nearest multiple of representation width
        const std::size_t repr_bytes = ext_int_representation_bytes(it->type());
        const std::size_t repr_bits = repr_bytes * config.ansi_c.char_width;

        const std::size_t unaligned_bits = w % repr_bits;
        const std::size_t pad = unaligned_bits ? repr_bits - unaligned_bits : 0;
        it = pad_ext_int_after(components, it, pad);
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
    else if(it->get_is_padding() && it_type.get_bool("#extint"))
    {
      // The alignment offset of ExtInt padding (that is not part of a bit field)
      // is accounted for by the main ExtInt field, so not done here
      assert(bit_field_bits == 0);
      continue;
    }
    else
      a = alignment(it_type, ns);

    assert(bit_field_bits == 0);
    assert(a > 0);

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

    if(it_type.get_bool("#extint"))
    {
      assert(!it->get_is_padding());
      std::size_t w = string2integer(it_type.width().as_string()).to_uint64();

      // If the next field is padding for this one, add its width to the offset
      // too
      const auto pad_field = std::next(it);
      if(
        pad_field != components.end() && pad_field->get_is_padding() &&
        pad_field->type().get_bool("#extint"))
      {
        w += string2integer(pad_field->type().width().as_string()).to_uint64();
      }

      assert(w % (a.to_uint64() * config.ansi_c.char_width) == 0);
      offset += w / config.ansi_c.char_width;
      continue;
    }

    type2tc thetype = migrate_type(it_type);
    offset += type_byte_size(ns.follow(thetype, true));
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
    size_bits = std::max(size_bits, type_byte_size_bits(thetype));
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
