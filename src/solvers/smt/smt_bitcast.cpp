#include <solvers/smt/smt_conv.h>

static expr2tc flatten_to_bitvector_rec(const expr2tc &new_expr)
{
  // Easy case, no need to concat anything
  if(is_number_type(new_expr) || is_pointer_type(new_expr))
    return new_expr;

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

    // First element
    expr2tc expr =
      index2tc(arraytype.subtype, new_expr, constant_int2tc(index_type2(), 0));

    // Concat elements if there are more than 1
    for(unsigned int i = 1; i < intref.value.to_uint64(); i++)
    {
      expr2tc tmp = index2tc(
        arraytype.subtype, new_expr, constant_int2tc(index_type2(), 0));
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

    // Concat elements if there are more than 1
    for(int i = sz - 2; i >= 0; i--)
    {
      expr2tc tmp =
        member2tc(structtype.members[i], new_expr, structtype.member_names[i]);
      type2tc res_type =
        get_uint_type(expr->type->get_width() + tmp->type->get_width());
      expr = concat2tc(res_type, expr, tmp);
    }

    return expr;
  }

  std::cerr << "Unrecognized type " << get_type_id(*new_expr->type);
  std::cerr << " when flattening to bytes" << std::endl;
  abort();
}

static expr2tc flatten_to_bitvector(const expr2tc &new_expr)
{
  const expr2tc concated_expr = flatten_to_bitvector_rec(new_expr);
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

    // Converting from struct to fp, we simply convert the struct to bv and use
    // the bv to fp method to do the job for us
    if(is_struct_type(new_from))
      new_from = flatten_to_bitvector(new_from);

    // from bitvectors should go through the fp api
    if(is_bv_type(new_from))
      return fp_api->mk_from_bv_to_fp(
        convert_ast(new_from), convert_sort(to_type));

    if(is_union_type(new_from))
    {
      std::cerr << "Unions not supported when bitcasting to fp for now\n";
      expr->dump();
      abort();
    }
  }

  if(is_bv_type(to_type))
  {
    unsigned int sz = expr->type->get_width() - from->type->get_width();
    if(is_floatbv_type(from))
      return is_signedbv_type(expr->type)
               ? mk_sign_ext(fp_api->mk_from_fp_to_bv(convert_ast(from)), sz)
               : mk_zero_ext(fp_api->mk_from_fp_to_bv(convert_ast(from)), sz);

    if(is_struct_type(from))
    {
      expr2tc new_bv = flatten_to_bitvector(from);
      if(!sz)
        return convert_ast(new_bv);

      return is_signedbv_type(to_type) ? mk_sign_ext(convert_ast(new_bv), sz)
                                       : mk_zero_ext(convert_ast(new_bv), sz);
    }

    if(is_union_type(from))
    {
      std::cerr << "Unions not supported when bitcasting to bv for now\n";
      expr->dump();
      abort();
    }
  }

  if(is_struct_type(to_type))
  {
    expr2tc new_from = from;

    // Converting from fp to struct, we simply convert the fp to bv and use
    // the bv to struct method to do the job for us
    if(is_floatbv_type(new_from))
      new_from =
        bitcast2tc(get_uint_type(new_from->type->get_width()), new_from);

    if(is_bv_type(new_from))
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
          extract2tc(structtype.members[i], new_from, offset + sz - 1, offset);
        fields.push_back(tmp);
      }

      return convert_ast(constant_struct2tc(to_type, fields));
    }

    if(is_union_type(from))
    {
      std::cerr << "Unions not supported when bitcasting to struct for now\n";
      expr->dump();
      abort();
    }
  }

  // Cast by value is fine
  return convert_ast(typecast2tc(to_type, from));
}
