// Danger Will Robinson: this is not a C++ class, but in fact a template, and
// is included by array_conv.h directly so that uses of it are instanciated
// correctly.

#include <ansi-c/c_types.h>

template <class subclass>
array_convt<subclass>::array_convt(bool enable_cache, bool int_encoding,
                         const namespacet &_ns, bool is_cpp, bool tuple_support)
  // Declare that we can put bools in arrays, and init unbounded arrays
  // XXX - can't put bools in arrays /just/ yet due to some type hiccups.
  : smt_convt(enable_cache, int_encoding, _ns, is_cpp, tuple_support, true,
              true), array_indexes(), array_values(), array_updates()
{
}

template <class subclass>
array_convt<subclass>::~array_convt()
{
}

template <class subclass>
const smt_ast *
array_convt<subclass>::convert_array_equality(const expr2tc &a, const expr2tc &b)
{

  // Only support a scenario where the lhs (a) is a symbol.
  assert(is_symbol2t(a) && "Malformed array equality");

  const array_ast *value;

  // We want everything to go through the expression cache. Except when creating
  // new arrays with either constant_array_of or constant_array.
  if (is_constant_expr(b)) {
    value = array_downcast(array_create(b));
  } else {
    value = array_downcast(convert_ast(b));
  }

  const symbol2t &sym = to_symbol2t(a);
  std::string symname = sym.get_symbol_name();

  assign_array_symbol(symname, value);

  // Also pump that into the smt cache.
  typename subclass::smt_cache_entryt e = { a, value, this->ctx_level };
  this->smt_cache.insert(e);

  // Return a true value, because that assignment is always true.
  return convert_ast(true_expr);
}

template <class subclass>
smt_ast *
array_convt<subclass>::fresh_array(const smt_sort *ms, const std::string &name)
{
  // No solver representation for this.
  unsigned long domain_width = ms->get_domain_width();
  unsigned long array_size = 1UL << domain_width;
  const smt_sort *range_sort =
    mk_sort(SMT_SORT_BV, ms->get_range_width(), false);

  array_ast *mast = new array_ast(ms);
  mast->symname = name;
  assign_array_symbol(name, mast);

  if (is_unbounded_array(mast->sort)) {
    // Don't attempt to initialize. Store the fact that we've allocated a
    // fresh new array.
    mast->base_array_id = array_indexes.size();
    mast->array_update_num = 0;
    std::set<expr2tc> tmp_set;
    array_indexes.push_back(tmp_set);

    std::vector<std::list<struct array_select> > tmp2;
    array_values.push_back(tmp2);

    std::list<struct array_select> tmp25;
    array_values[mast->base_array_id].push_back(tmp25);

    std::vector<struct array_with> tmp3;
    array_updates.push_back(tmp3);

    // Aimless piece of data, just to keep indexes in iarray_updates and
    // array_values in sync.
    struct array_with w;
    w.is_ite = false;
    w.idx = expr2tc();
    array_updates[mast->base_array_id].push_back(w);

    array_subtypes.push_back(ms->get_range_width());

    return mast;
  }

  mast->array_fields.reserve(array_size);

  // Populate that array with a bunch of fresh bvs of the correct sort.
  unsigned long i;
  for (i = 0; i < array_size; i++) {
    const smt_ast *a = mk_fresh(range_sort, "array_fresh_array::");
    mast->array_fields.push_back(a);
  }

  return mast;
}

template <class subclass>
const smt_ast *
array_convt<subclass>::mk_select(const expr2tc &array, const expr2tc &idx,
                         const smt_sort *ressort)
{
  assert(ressort->id != SMT_SORT_ARRAY);
  const array_ast *ma = array_downcast(convert_ast(array));

  if (is_unbounded_array(ma->sort))
    return mk_unbounded_select(ma, idx, ressort);

  assert(ma->array_fields.size() != 0);

  // If this is a constant index, simple. If not, not.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.constant_value.to_ulong();
    if (intval > ma->array_fields.size())
      // Return a fresh value.
      return mk_fresh(ressort, "array_mk_select_badidx::");

    // Otherwise,
    return ma->array_fields[intval];
  }

  // What we have here is a nondeterministic index. Alas, compare with
  // everything.
  const smt_ast *fresh = mk_fresh(ressort, "array_mk_select::");
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *args[2], *idxargs[2], *impargs[2];
  unsigned long dom_width = ma->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  args[0] = fresh;
  idxargs[0] = real_idx;

  for (unsigned long i = 0; i < ma->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = this->mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);
    args[1] = ma->array_fields[i];
    const smt_ast *val_eq = this->mk_func_app(bool_sort, SMT_FUNC_EQ, args, 2);

    impargs[0] = idx_eq;
    impargs[1] = val_eq;

    this->assert_lit(mk_lit(this->mk_func_app(bool_sort, SMT_FUNC_IMPLIES, impargs, 2)));
  }

  return fresh;
}

template <class subclass>
const smt_ast *
array_convt<subclass>::mk_store(const expr2tc &array, const expr2tc &idx,
                        const expr2tc &value, const smt_sort *ressort)
{
  const array_ast *ma = array_downcast(convert_ast(array));

  if (is_unbounded_array(ma->sort))
    return mk_unbounded_store(ma, idx, convert_ast(value), ressort);

  assert(ma->array_fields.size() != 0);

  array_ast *mast =
    new array_ast(ressort, ma->array_fields);

  // If this is a constant index, simple. If not, not.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.constant_value.to_ulong();
    if (intval > ma->array_fields.size())
      return convert_ast(array);

    // Otherwise,
    mast->array_fields[intval] = convert_ast(value);
    return mast;
  }

  // Oh dear. We need to update /all the fields/ :(
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *real_value = convert_ast(value);
  const smt_ast *iteargs[3], *idxargs[2];
  unsigned long dom_width = mast->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  idxargs[0] = real_idx;
  iteargs[1] = real_value;

  for (unsigned long i = 0; i < mast->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = this->mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);

    iteargs[0] = idx_eq;
    iteargs[2] = mast->array_fields[i];

    const smt_ast *new_val =
      this->mk_func_app(iteargs[1]->sort, SMT_FUNC_ITE, iteargs, 3);
    mast->array_fields[i] = new_val;
  }

  return mast;
}

template <class subclass>
const smt_ast *
array_convt<subclass>::mk_unbounded_select(const array_ast *ma,
                                   const expr2tc &real_idx,
                                   const smt_sort *ressort)
{
  // Record that we've accessed this index.
  array_indexes[ma->base_array_id].insert(real_idx);

  // Generate a new free variable
  smt_ast *a = mk_fresh(ressort, "mk_unbounded_select");

  struct array_select sel;
  sel.src_array_update_num = ma->array_update_num;
  sel.idx = real_idx;
  sel.val = a;
  // Record this index
  array_values[ma->base_array_id][ma->array_update_num].push_back(sel);

  // Convert index; it might trigger an array_of, or something else, which
  // fiddles with other arrays.
  convert_ast(real_idx);

  return a;
}

template <class subclass>
const smt_ast *
array_convt<subclass>::mk_unbounded_store(const array_ast *ma,
                                  const expr2tc &idx, const smt_ast *value,
                                  const smt_sort *ressort)
{
  // Record that we've accessed this index.
  array_indexes[ma->base_array_id].insert(idx);

  // More nuanced: allocate a new array representation.
  array_ast *newarr = new array_ast(ressort);
  newarr->base_array_id = ma->base_array_id;
  newarr->array_update_num = array_updates[ma->base_array_id].size();

  // Record update
  struct array_with w;
  w.is_ite = false;
  w.idx = idx;
  w.u.w.src_array_update_num = ma->array_update_num;
  w.u.w.val = value;
  array_updates[ma->base_array_id].push_back(w);

  // Convert index; it might trigger an array_of, or something else, which
  // fiddles with other arrays.
  convert_ast(idx);

  // Also file a new select record for this point in time.
  std::list<struct array_select> tmp;
  array_values[ma->base_array_id].push_back(tmp);

  // Result is the new array id goo.
  return newarr;
}

template <class subclass>
smt_ast *
array_convt<subclass>::array_ite(const smt_ast *_cond,
                         const smt_ast *_true_arr,
                         const smt_ast *_false_arr,
                         const smt_sort *thesort)
{
  const array_ast *cond = array_downcast(_cond);
  const array_ast *true_arr = array_downcast(_true_arr);
  const array_ast *false_arr = array_downcast(_false_arr);

  if (is_unbounded_array(true_arr->sort))
    return unbounded_array_ite(cond, true_arr, false_arr, thesort);

  // For each element, make an ite.
  assert(true_arr->array_fields.size() != 0 &&
         true_arr->array_fields.size() == false_arr->array_fields.size());
  array_ast *mast = new array_ast(thesort);
  const smt_ast *args[3];
  args[0] = cond;
  unsigned long i;
  for (i = 0; i < true_arr->array_fields.size(); i++) {
    // One ite pls.
    args[1] = true_arr->array_fields[i];
    args[2] = false_arr->array_fields[i];
    const smt_ast *res = this->mk_func_app(args[1]->sort, SMT_FUNC_ITE, args, 3);
    mast->array_fields.push_back(array_downcast(res));
  }

  return mast;
}

template <class subclass>
smt_ast *
array_convt<subclass>::unbounded_array_ite(const array_ast *cond,
                                   const array_ast *true_arr,
                                   const array_ast *false_arr,
                                   const smt_sort *thesort)
{
  // Precondition for a lot of goo: that the two arrays are the same, at
  // different points in time.
  assert(true_arr->base_array_id == false_arr->base_array_id &&
         "ITE between two arrays with different bases are unsupported");

  array_ast *newarr = new array_ast(thesort);
  newarr->base_array_id = true_arr->base_array_id;
  newarr->array_update_num = array_updates[true_arr->base_array_id].size();

  struct array_with w;
  w.is_ite = true;
  w.idx = expr2tc();
  w.u.i.src_array_update_true = true_arr->array_update_num;
  w.u.i.src_array_update_false = false_arr->array_update_num;
  w.u.i.cond = cond;
  array_updates[true_arr->base_array_id].push_back(w);

  // Also file a new select record for this point in time.
  std::list<struct array_select> tmp;
  array_values[true_arr->base_array_id].push_back(tmp);

  return newarr;
}

template <class subclass>
const smt_ast *
array_convt<subclass>::convert_array_of(const expr2tc &init_val,
                                unsigned long domain_width)
{
  const smt_sort *dom_sort = mk_sort(SMT_SORT_BV, domain_width, false);
  const smt_sort *idx_sort = convert_sort(init_val->type);

  if (!this->int_encoding && is_bool_type(init_val) && this->no_bools_in_arrays)
    idx_sort = mk_sort(SMT_SORT_BV, 1, false);

  const smt_sort *arr_sort = mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort);

  array_ast *mast = new array_ast(arr_sort);

  const smt_ast *init = convert_ast(init_val);
  if (!this->int_encoding && is_bool_type(init_val) && this->no_bools_in_arrays)
    init = make_bool_bit(init);

  if (is_unbounded_array(arr_sort)) {
    std::string name = mk_fresh_name("array_of_unbounded::");
    mast = static_cast<array_ast*>(fresh_array(arr_sort, name));
    array_of_vals.insert(std::pair<unsigned, const smt_ast *>
                                  (mast->base_array_id, init));
  } else {
    unsigned long array_size = 1UL << domain_width;
    for (unsigned long i = 0; i < array_size; i++)
      mast->array_fields.push_back(init);
  }

  return mast;
}

template <class subclass>
expr2tc
array_convt<subclass>::fixed_array_get(const smt_ast *a, const type2tc &type)
{
  const array_ast *mast = array_downcast(a);
  const array_type2t &arr = to_array_type(type);

  std::vector<expr2tc> fields;
  fields.reserve(mast->array_fields.size());
  for (unsigned int i = 0; i < mast->array_fields.size(); i++) {
    fields.push_back(get_bv(arr.subtype, mast->array_fields[i]));
  }

  constant_array2tc result(type, fields);
  return result;
}

template <class subclass>
expr2tc
array_convt<subclass>::array_get(const smt_ast *a, const type2tc &type)
{
  const smt_sort *s = convert_sort(type);
  if (!is_unbounded_array(s)) {
    return fixed_array_get(a, type);
  }

  const array_type2t &t = to_array_type(type);

  const array_ast *mast = array_downcast(a);

  if (mast->base_array_id >= array_valuation.size()) {
    // This is an array that was not previously converted, therefore doesn't
    // appear in the valuation table. Therefore, all its values are free.
    return expr2tc();
  }

  // Fetch all the indexes
  const std::set<expr2tc> &indexes = array_indexes[mast->base_array_id];

  std::map<expr2tc, unsigned> idx_map;
  for (std::set<expr2tc>::const_iterator it = indexes.begin();
       it != indexes.end(); it++)
    idx_map.insert(std::pair<expr2tc, unsigned>(*it, idx_map.size()));

  // Pick a set of array values.
  const std::vector<const smt_ast *> &solver_values =
    array_valuation[mast->base_array_id][mast->array_update_num];

  // Evaluate each index and each value.
  BigInt::ullong_t max_idx = 0;
  std::vector<std::pair<expr2tc, expr2tc> > values;
  values.resize(idx_map.size());
  for (std::map<expr2tc, unsigned>::const_iterator it = idx_map.begin();
       it != idx_map.end(); it++) {
    expr2tc idx = it->first;
    if (!is_constant_expr(idx))
      idx = this->get(idx);

    const smt_ast *this_value = solver_values[it->second];

    // Read the valuation. Guarenteed not to be an array or struct.
    assert((this_value->sort->id == SMT_SORT_BOOL ||
            this_value->sort->id == SMT_SORT_BV) &&
           "Unexpected sort in array field");

    // unimplemented
    expr2tc real_value;
    if (this_value->sort->id == SMT_SORT_BOOL)
      real_value = get_bool(this_value);
    else
      real_value = get_bv(t.subtype, this_value);

    values[it->second] = std::pair<expr2tc, expr2tc>(idx, real_value);

    // And record the largest index
    max_idx = std::max(max_idx,
                       to_constant_int2t(idx).constant_value.to_ulong());
  }

  // Work out the size of the array. If it's too large, clip it. Fill the
  // remaining elements with their values. This is lossy: if we want accuracy
  // in the future, then we need to find a way of returning sparse arrays
  // for potentially unbounded array sizes.
  if (max_idx > 1024)
    max_idx = 1024;

  type2tc arr_type(new array_type2t(t.subtype,
                                constant_int2tc(index_type2(), BigInt(max_idx)),
                                false));
  std::vector<expr2tc> array_values;
  array_values.resize(max_idx);
  for (unsigned int i = 0; i < values.size(); i++) {
    const constant_int2t &ref = to_constant_int2t(values[i].first);
    uint64_t this_idx = ref.constant_value.to_ulong();
    if (this_idx >= max_idx)
      continue;

    array_values[this_idx] = values[i].second;
  }

  return constant_array2tc(arr_type, array_values);
}

template <class subclass>
void
array_convt<subclass>::add_array_constraints(void)
{

  for (unsigned int i = 0; i < array_indexes.size(); i++) {
    add_array_constraints(i);
  }

  return;
}

template <class subclass>
void
array_convt<subclass>::add_array_constraints(unsigned int arr)
{
  // Right: we need to tie things up regarding these bitvectors. We have a
  // set of indexes...
  const std::set<expr2tc> &indexes = array_indexes[arr];

  // What we're going to build is a two-dimensional vector ish of each element
  // at each point in time. Expensive, but meh.
  array_valuation.resize(array_valuation.size() + 1);
  std::vector<std::vector<const smt_ast *> > &real_array_values =
    array_valuation.back();

  // Subtype is thus
  const smt_sort *subtype = mk_sort(SMT_SORT_BV, array_subtypes[arr], false);

  // Pre-allocate all the storage.
  real_array_values.resize(array_values[arr].size());
  for (unsigned int i = 0; i < real_array_values.size(); i++)
    real_array_values[i].resize(indexes.size());

  // Compute a mapping between indexes and an element in the vector. These
  // are ordered by how std::set orders them, not by history or anything. Or
  // even the element index.
  std::map<expr2tc, unsigned> idx_map;
  for (std::set<expr2tc>::const_iterator it = indexes.begin();
       it != indexes.end(); it++)
    idx_map.insert(std::pair<expr2tc, unsigned>(*it, idx_map.size()));

  assert(idx_map.size() == indexes.size());

  // Initialize the first set of elements.
  std::map<unsigned, const smt_ast*>::const_iterator it =
    array_of_vals.find(arr);
  if (it != array_of_vals.end()) {
    collate_array_values(real_array_values[0], idx_map, array_values[arr][0],
        subtype, it->second);
  } else {
    collate_array_values(real_array_values[0], idx_map, array_values[arr][0],
        subtype);
  }

  add_initial_ackerman_constraints(real_array_values[0], idx_map);

  // Now repeatedly execute transitions between states.
  for (unsigned int i = 0; i < real_array_values.size() - 1; i++)
    execute_array_trans(real_array_values, arr, i, idx_map, subtype);

}

template <class subclass>
void
array_convt<subclass>::execute_array_trans(
                            std::vector<std::vector<const smt_ast *> > &data,
                                   unsigned int arr,
                                   unsigned int idx,
                                   const std::map<expr2tc, unsigned> &idx_map,
                                   const smt_sort *subtype)
{
  // Steps: First, fill the destination vector with either free variables, or
  // the free variables that resulted for selects corresponding to that item.
  // Then apply update or ITE constraints.
  // Then apply equalities between the old and new values.

  std::vector<const smt_ast *> &dest_data = data[idx+1];
  collate_array_values(dest_data, idx_map, array_values[arr][idx+1], subtype);

  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  // Two updates that could have occurred for this array: a simple with, or
  // an ite.
  const array_with &w = array_updates[arr][idx+1];
  if (w.is_ite) {
    // Turn every index element into an ITE representing this operation. Every
    // single element is addressed and updated; no further constraints are
    // needed. Not even the ackerman ones, in fact, because instances of that
    // from previous array updates will feed through to here (speculation).

    unsigned int true_idx = w.u.i.src_array_update_true;
    unsigned int false_idx = w.u.i.src_array_update_false;
    assert(true_idx < idx + 1 && false_idx < idx + 1);
    const std::vector<const smt_ast *> &true_vals = data[true_idx];
    const std::vector<const smt_ast *> &false_vals = data[false_idx];
    const smt_ast *cond = w.u.i.cond;

    // Each index value becomes an ITE between each source value.
    const smt_ast *args[3], *eq[2];
    args[0] = cond;
    for (unsigned int i = 0; i < idx_map.size(); i++) {
      args[1] = true_vals[i];
      args[2] = false_vals[i];
      eq[0] = this->mk_func_app(subtype, SMT_FUNC_ITE, args, 3);
      eq[1] = dest_data[i];
      this->assert_lit(mk_lit(this->mk_func_app(boolsort, SMT_FUNC_EQ, eq, 2)));
    }
  } else {
    // Place a constraint on the updated variable; add equality constraints
    // between the older version and this version.

    // So, the updated element,
    std::map<expr2tc, unsigned>::const_iterator it = idx_map.find(w.idx);
    assert(it != idx_map.end());

    const expr2tc &update_idx_expr = it->first;
    const smt_ast *update_idx_ast = convert_ast(update_idx_expr);
    unsigned int updated_idx = it->second;
    const smt_ast *updated_value = w.u.w.val;

    // Assign in its value.
    dest_data[updated_idx] = updated_value;

    // Check all the values selected out of this instance; if any have the same
    // index, tie the select's fresh variable to the updated value. If there are
    // differing index exprs that evaluate to the same location they'll be
    // caught by code later.
    const std::list<struct array_select> &sels = array_values[arr][idx+1];
    for (typename std::list<struct array_select>::const_iterator it = sels.begin();
         it != sels.end(); it++) {
      if (it->idx == update_idx_expr) {
        const smt_ast *args[2];
        args[0] = updated_value;
        args[1] = it->val;
        this->assert_lit(mk_lit(this->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    }

    // Now look at all those other indexes...
    assert(w.u.w.src_array_update_num < idx+1);
    const std::vector<const smt_ast *> &source_data =
      data[w.u.w.src_array_update_num];

    const smt_ast *args[3];
    unsigned int i = 0;
    for (std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.begin();
         it2 != idx_map.end(); it2++, i++) {
      if (it2->second == updated_idx)
        continue;

      // Generate an ITE. If the index is nondeterministically equal to the
      // current index, take the updated value, otherwise the original value.
      // This departs from the CBMC implementation, in that they explicitly
      // use implies and ackerman constraints.
      // FIXME: benchmark the two approaches. For now, this is shorter.
      args[0] = update_idx_ast;
      args[1] = convert_ast(it2->first);
      args[0] = this->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      args[1] = updated_value;
      args[2] = source_data[i];
      args[0] = this->mk_func_app(subtype, SMT_FUNC_ITE, args, 3);
      args[1] = dest_data[i];
      this->assert_lit(mk_lit(this->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      // The latter part of this could be replaced with more complex logic,
      // that only asserts an equality between selected values, and just stores
      // the result of the ITE for all other values. FIXME: try this.
    }
  }
}

template <class subclass>
void
array_convt<subclass>::collate_array_values(std::vector<const smt_ast *> &vals,
                                    const std::map<expr2tc, unsigned> &idx_map,
                                    const std::list<struct array_select> &idxs,
                                    const smt_sort *subtype,
                                    const smt_ast *init_val)
{
  // So, the value vector should be allocated but not initialized,
  assert(vals.size() == idx_map.size());

  // First, make everything null,
  for (std::vector<const smt_ast *>::iterator it = vals.begin();
       it != vals.end(); it++)
    *it = NULL;

  // Now assign in all free variables created as a result of selects.
  for (typename std::list<struct array_select>::const_iterator it = idxs.begin();
       it != idxs.end(); it++) {
    std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.find(it->idx);
    assert(it2 != idx_map.end());
    vals[it2->second] = it->val;
  }

  // Initialize everything else to either a free variable or the initial value.
  if (init_val == NULL) {
    // Free variables, except where free variables tied to selects have occurred
    for (std::vector<const smt_ast *>::iterator it = vals.begin();
         it != vals.end(); it++) {
      if (*it == NULL)
        *it = mk_fresh(subtype, "collate_array_vals::");
    }
  } else {
    // We need to assign the initial value in, except where there's already
    // a select/index, in which case we assert that the values are equal.
    const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
    for (std::vector<const smt_ast *>::iterator it = vals.begin();
         it != vals.end(); it++) {
      if (*it == NULL) {
        *it = init_val;
      } else {
        const smt_ast *args[2];
        args[0] = *it;
        args[1] = init_val;
        this->assert_lit(mk_lit(this->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    }
  }

  // Fin.
}

template <class subclass>
void
array_convt<subclass>::add_initial_ackerman_constraints(
                                  const std::vector<const smt_ast *> &vals,
                                  const std::map<expr2tc,unsigned> &idx_map)
{
  // Lolquadratic,
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  for (std::map<expr2tc, unsigned>::const_iterator it = idx_map.begin();
       it != idx_map.end(); it++) {
    const smt_ast *outer_idx = convert_ast(it->first);
    for (std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.begin();
         it2 != idx_map.end(); it2++) {
      const smt_ast *inner_idx = convert_ast(it2->first);

      // If they're the same idx, they're the same value.
      const smt_ast *args[2];
      args[0] = outer_idx;
      args[1] = inner_idx;
      const smt_ast *idxeq = this->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

      args[0] = vals[it->second];
      args[1] = vals[it2->second];
      const smt_ast *valeq = this->mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

      args[0] = idxeq;
      args[1] = valeq;
      this->assert_lit(mk_lit(this->mk_func_app(boolsort, SMT_FUNC_IMPLIES, args, 2)));
    }
  }
}
