#include <solvers/smt/smt_conv.h>
#include <util/type_byte_size.h>

/**
 * Constructs the tree-like concatenation of expressions from a sequence.
 *
 * Invokes `extract` for each index in [start,start+n) and concatenates the
 * results to one expression, which is is returned. The expression forms a
 * binary tree of minimal height with the `extract(i)` expressions at its
 * leaves and `concat2t` expressions otherwise.
 *
 * For each valid index `i` in the above range, `extract(i)` should return the
 * `i`ths sub-expression to concatenate.
 *
 * @param start   The initial index to invoke `extract` for
 * @param n       The number of successive elements to extract starting at
 *                `start`; note: n > 0 only
 * @param extract Callback to invoke for each valid index
 *
 * @return An expression corresponding to the concatenation (in order, from
 *         `start` to `start+n-1`) of the `extract` results and its size
 */
template <typename Extract>
static expr2tc concat_tree(size_t start, size_t n, const Extract &extract)
{
  assert(n);
  if(n == 1)
    return extract(start);

  /* here, n > 1: recursively build 2 sub-expressions to concatenate, both of
   * similar depth logarithmic in n to avoid a stack overflow in convert_ast()
   * down the line when n is large, for instance in #732 case 2.
   *
   * We could also return the size along with the expression in order to
   * avoid unnecessarily re-computing it in this recursion by calling
   * type_byte_size_bits() on the exprs for both branches: both results are
   * already known and available. When `extract` operates on an array, its
   * subtype's size indeed only would need to be computed once, regardless of
   * `n`. However, I've not been able to measure performance benefits as the
   * dynamic allocations `extract` usually performs dwarf the size computation.
   */
  expr2tc a = concat_tree(start, n / 2, extract);
  expr2tc b = concat_tree(start + n / 2, n - n / 2, extract);
  size_t sz = type_byte_size_bits(a->type).to_uint64() +
              type_byte_size_bits(b->type).to_uint64();
  return concat2tc(get_uint_type(sz), a, b);
}

static expr2tc flatten_to_bitvector(const expr2tc &new_expr)
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

    size_t sz = intref.value.to_uint64();
    type2tc idx = index_type2();

    auto extract = [&](size_t i) {
      /* The sub-expression should be flattened as well */
      return flatten_to_bitvector(index2tc(
        arraytype.subtype, new_expr, constant_int2tc(idx, sz - i - 1)));
    };

    return concat_tree(0, sz, extract);
  }

  if(new_expr->type->get_width() == 0)
    return constant_int2tc(get_uint_type(0), BigInt(0));

  // If it is a struct, concat all members into a big bitvector
  // TODO: this is similar to concat array elements, should we merge them?
  if(is_struct_type(new_expr))
  {
    const struct_type2t &structtype = to_struct_type(new_expr->type);

    size_t sz = structtype.members.size();

    // Iterate over each member and flatten them

    auto extract = [&](size_t i) {
      /* The sub-expression should be flattened as well */
      log_debug("[{}, {}] Creating member", __FILE__, __LINE__);
      return flatten_to_bitvector(member2tc(
        structtype.members[sz - i - 1],
        new_expr,
        structtype.member_names[sz - i - 1]));
    };

    return concat_tree(0, sz, extract);
  }

  log_error("Unrecognized type {} when flattening to bytes",
    get_type_id(*new_expr->type));
  abort();
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
      new_from = flatten_to_bitvector(new_from);

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
      return convert_ast(flatten_to_bitvector(from));

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
      new_from = flatten_to_bitvector(new_from);

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
  else if(is_array_type(to_type))
  {
    expr2tc new_from = from;

    if(is_floatbv_type(new_from))
      new_from =
        bitcast2tc(get_uint_type(new_from->type->get_width()), new_from);

    // Converting from struct to array, we convert it to bv and use the bv to
    // struct method to do the job for us
    if(is_struct_type(new_from))
      new_from = flatten_to_bitvector(new_from);

    if(is_bv_type(new_from) || is_union_type(new_from))
    {
      // TODO: handle multidimensional arrays
      assert(
        !is_multi_dimensional_array(to_type) &&
        "Bitcasting to multidimensional arrays is not supported for now\n");

      array_type2t arr_type = to_array_type(to_type);
      type2tc subtype = arr_type.subtype;

      // We shouldn't have any bit left behind
      assert(new_from->type->get_width() % subtype->get_width() == 0);
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
