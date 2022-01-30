#include <solvers/smt/smt_conv.h>
#include <util/type_byte_size.h>

static expr2tc
flatten_to_bitvector_rec(const expr2tc &new_expr, const messaget &msg)
{
  // Easy case, no need to concat anything
  if(is_number_type(new_expr))
    return new_expr;

  if(is_pointer_type(new_expr))
    return bitcast2tc(get_uint_type(new_expr->type->get_width()), new_expr);

  // If it is an array, concat every element into a big bitvector
  if(is_array_type(new_expr))
  {
    // Assume only fixed-size arrays
    const array_type2t &arraytype = to_array_type(new_expr->type);
    assert(
      !arraytype.size_is_infinite && !is_nil_expr(arraytype.array_size) &&
      is_constant_int2t(arraytype.array_size) &&
      "Can't flatten array with unbounded size");

    // Iterate over each element and flatten them
    const constant_int2t &intref = to_constant_int2t(arraytype.array_size);
    assert(intref.value > 0);

    int sz = intref.value.to_uint64();

    // First element
    expr2tc expr = index2tc(
      arraytype.subtype, new_expr, constant_int2tc(index_type2(), sz - 1));
    expr = flatten_to_bitvector_rec(expr, msg);

    // Concat elements if there are more than 1
    for(int i = sz - 2; i >= 0; i--)
    {
      expr2tc tmp = index2tc(
        arraytype.subtype, new_expr, constant_int2tc(index_type2(), i));
      tmp = flatten_to_bitvector_rec(tmp, msg);
      type2tc res_type =
        get_uint_type(expr->type->get_width() + tmp->type->get_width());
      expr = concat2tc(res_type, expr, tmp);
    }

    return expr;
  }

  // If it is an array, concat every member into a big bitvector
  // TODO: this is similar to concat array elements, should we merge them?
  if(is_struct_type(new_expr))
  {
    const struct_type2t &structtype = to_struct_type(new_expr->type);

    int sz = structtype.members.size();

    // Iterate over each member and flatten them
    expr2tc expr = member2tc(
      structtype.members[sz - 1], new_expr, structtype.member_names[sz - 1]);
    expr = flatten_to_bitvector_rec(expr, msg);

    // Concat elements if there are more than 1
    for(int i = sz - 2; i >= 0; i--)
    {
      expr2tc tmp =
        member2tc(structtype.members[i], new_expr, structtype.member_names[i]);
      tmp = flatten_to_bitvector_rec(tmp, msg);
      type2tc res_type =
        get_uint_type(expr->type->get_width() + tmp->type->get_width());
      expr = concat2tc(res_type, expr, tmp);
    }

    return expr;
  }

  if(is_union_type(new_expr))
  {
    bool big_endian =
      config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN;

    expr2tc expr = byte_extract2tc(
      get_uint8_type(),
      new_expr,
      constant_int2tc(index_type2(), 0),
      big_endian);
    expr = flatten_to_bitvector_rec(expr, msg);

    // Concat elements if there are more than 1
    BigInt size = type_byte_size(new_expr->type);
    for(int i = 1; i < size; i++)
    {
      expr2tc tmp = byte_extract2tc(
        get_uint8_type(),
        new_expr,
        constant_int2tc(index_type2(), i),
        big_endian);
      tmp = flatten_to_bitvector_rec(tmp, msg);
      type2tc res_type =
        get_uint_type(expr->type->get_width() + tmp->type->get_width());
      expr = concat2tc(res_type, expr, tmp);
    }

    return expr;
  }

  msg.error(fmt::format(
    "Unrecognized type {} when flattening to bytes",
    get_type_id(*new_expr->type)));
  abort();
}

static expr2tc
flatten_to_bitvector(const expr2tc &new_expr, const messaget &msg)
{
  const expr2tc concated_expr = flatten_to_bitvector_rec(new_expr, msg);
  return concated_expr;
}

smt_astt smt_convt::convert_bitcast(const expr2tc &expr)
{
  assert(is_bitcast2t(expr));

  const expr2tc &from = to_bitcast2t(expr).from;
  const type2tc &to_type = to_bitcast2t(expr).type;

  // Converts to floating-point
  if(is_floatbv_type(to_type))
  {
    expr2tc new_from = from;

    // Converting from struct/array to fp, we simply convert it to bv and use
    // the bv to fp method to do the job for us
    if(is_struct_type(new_from) || is_array_type(new_from))
      new_from = flatten_to_bitvector(new_from, msg);

    // from bitvectors should go through the fp api
    if(is_bv_type(new_from) || is_union_type(new_from))
      return fp_api->mk_from_bv_to_fp(
        convert_ast(new_from), convert_sort(to_type));
  }
  else if(is_bv_type(to_type))
  {
    if(is_floatbv_type(from))
      return fp_api->mk_from_fp_to_bv(convert_ast(from));

    if(is_struct_type(from) || is_array_type(from))
      return convert_ast(flatten_to_bitvector(from, msg));

    if(is_union_type(from))
      return convert_ast(from);
  }
  else if(is_struct_type(to_type))
  {
    expr2tc new_from = from;

    // Converting from fp to struct, we simply convert the fp to bv and use
    // the bv to struct method to do the job for us
    if(is_floatbv_type(new_from))
      new_from =
        bitcast2tc(get_uint_type(new_from->type->get_width()), new_from);

    // Converting from array to struct, we convert it to bv and use the bv to
    // struct method to do the job for us
    if(is_array_type(new_from))
      new_from = flatten_to_bitvector(new_from, msg);

    if(is_bv_type(new_from) || is_union_type(new_from))
    {
      const struct_type2t &structtype = to_struct_type(to_type);

      // We have to reconstruct the struct from the bitvector, so do it
      // by extracting the offsets+size of each member from the bitvector
      std::vector<expr2tc> fields;
      for(unsigned int i = 0; i < structtype.members.size(); i++)
      {
        unsigned int offset =
          member_offset_bits(to_type, structtype.member_names[i]).to_uint64();
        unsigned int sz =
          type_byte_size_bits(structtype.members[i]).to_uint64();
        expr2tc tmp =
          extract2tc(get_uint_type(sz), new_from, offset + sz - 1, offset);
        fields.push_back(bitcast2tc(structtype.members[i], tmp));
      }

      return convert_ast(constant_struct2tc(to_type, fields));
    }
  }
  else if(is_union_type(to_type))
  {
    if(is_bv_type(from))
      return convert_ast(from);
  }
  else if(is_array_type(to_type))
  {
    expr2tc new_from = from;

    if(is_floatbv_type(new_from))
      new_from =
        bitcast2tc(get_uint_type(new_from->type->get_width()), new_from);

    // Converting from struct to array, we convert it to bv and use the bv to
    // struct method to do the job for us
    if(is_struct_type(new_from))
      new_from = flatten_to_bitvector(new_from, msg);

    if(is_bv_type(new_from) || is_union_type(new_from))
    {
      // TODO: handle multidimensional arrays
      assert(
        !is_multi_dimensional_array(to_type) &&
        "Bitcasting to multidimensional arrays is not supported for now\n");

      array_type2t arr_type = to_array_type(to_type);
      type2tc subtype = arr_type.subtype;

      // We shouldn't have any bit left behind
      // This will not work for FAMs!
      //assert(new_from->type->get_width() % subtype->get_width() == 0);
      unsigned int num_el = new_from->type->get_width() / subtype->get_width();

      std::vector<expr2tc> elems;
      for(unsigned int i = 0; i < num_el; ++i)
      {
        unsigned int sz = subtype->get_width();
        unsigned int offset = i * sz;
        expr2tc tmp = extract2tc(
          get_uint_type(subtype->get_width()),
          new_from,
          offset + sz - 1,
          offset);
        elems.push_back(bitcast2tc(subtype, tmp));
      }

      return convert_ast(constant_array2tc(to_type, elems));
    }
  }

  // Cast by value is fine
  return convert_ast(typecast2tc(to_type, from));
}
