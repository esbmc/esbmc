#include <algorithm>
#include <set>
#include <solvers/smt/array_conv.h>
#include <util/c_types.h>
#include <utility>

static inline bool
array_indexes_are_same(
    const array_convt::idx_record_containert &a,
    const array_convt::idx_record_containert &b)
{
  if (a.size() != b.size())
    return false;

  for (const auto &e : a) {
    if (b.find(e.idx) == b.end())
      return false;
  }

  return true;
}

array_convt::array_convt(smt_convt *_ctx) : array_iface(true, true),
  array_indexes(), array_selects(), array_updates(), ctx(_ctx)
{
}

array_convt::~array_convt()
{
}

void
array_convt::convert_array_assign(const array_ast *src, smt_astt sym)
{

  // Implement array assignments by simply making the destination AST track the
  // same array. No new variables need be introduced, saving lots of searching
  // hopefully. This works because we're working with an SSA program where the
  // source array will never be modified.

  // Get a mutable reference to the destination
  array_ast *destination = const_cast<array_ast*>(array_downcast(sym));
  const array_ast *source = src;

  // And copy across it's valuation
  destination->array_fields = source->array_fields;
  destination->base_array_id = source->base_array_id;
  destination->array_update_num = source->array_update_num;
  return;
}

unsigned int
array_convt::new_array_id(void)
{
  unsigned int new_base_array_id = array_indexes.size();

  // Pouplate tracking data with empt containers
  idx_record_containert tmp_set;
  array_indexes.push_back(tmp_set);

  array_select_containert tmp2;
  array_selects.push_back(tmp2);

  array_update_containert tmp3;
  array_updates.push_back(tmp3);

  index_map_containert tmp4;
  expr_index_map.push_back(tmp4);

  // Aimless piece of data, just to keep indexes in iarray_updates and
  // array_selects in sync.
  struct array_with w;
  w.is_ite = false;
  w.idx = expr2tc();
  w.ctx_level = UINT_MAX; // ahem
  w.update_level = 0;
  array_updates[new_base_array_id].insert(w);

  touched_array_sett touched;

  // Insert self into 'touched' arrays
  touched_arrayt t;
  t.array_id = new_base_array_id;
  t.ctx_level = ctx->ctx_level;
  touched.insert(t);

  array_relations.push_back(touched);

  array_valuation.push_back(array_update_vect());
  array_valuation.back().push_back(ast_vect());

  return new_base_array_id;
}

smt_ast *
array_convt::mk_array_symbol(const std::string &name, smt_sortt ms,
                             smt_sortt subtype)
{
  assert(subtype->id != SMT_SORT_ARRAY && "Can't create array of arrays with "
         "array flattener. Should be flattened elsewhere");

  // Create either a new bounded or unbounded array.
  unsigned long domain_width = ms->domain_width;
  unsigned long array_size = 1UL << domain_width;

  // Create new AST storage
  array_ast *mast = new_ast(ms);
  mast->symname = name;

  if (is_unbounded_array(mast->sort)) {
    // Don't attempt to initialize: this array is of unbounded size. Instead,
    // record a fresh new array.

    // Array ID: identifies an array at a level that corresponds to 'level1'
    // renaming, or having storage in C. Accumulates a history of selects and
    // updates.
    mast->base_array_id = new_array_id();
    mast->array_update_num = 0;

    array_subtypes.push_back(subtype);

    return mast;
  }

  // For bounded arrays, populate it's storage vector with a bunch of fresh bvs
  // of the correct sort.
  mast->array_fields.reserve(array_size);

  unsigned long i;
  for (i = 0; i < array_size; i++) {
    smt_astt a = ctx->mk_fresh(subtype, "array_fresh_array::");
    mast->array_fields.push_back(a);
  }

  return mast;
}

smt_astt
array_convt::mk_select(const array_ast *ma, const expr2tc &idx,
                         smt_sortt ressort)
{

  // Create a select: either hand off to the unbounded implementation, or
  // continue for bounded-size arrays
  if (is_unbounded_array(ma->sort))
    return mk_unbounded_select(ma, idx, ressort);

  assert(ma->array_fields.size() != 0);

  // If this is a constant index, then simply access the designated element.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.value.to_ulong();
    if (intval > ma->array_fields.size())
      // Return a fresh value.
      return ctx->mk_fresh(ressort, "array_mk_select_badidx::");

    // Otherwise,
    return ma->array_fields[intval];
  }

  // For undetermined indexes, create a large case switch across all values.
  smt_astt fresh = ctx->mk_fresh(ressort, "array_mk_select::");
  smt_astt real_idx = ctx->convert_ast(idx);
  unsigned long dom_width = ma->sort->domain_width;
  smt_sortt bool_sort = ctx->boolean_sort;

  for (unsigned long i = 0; i < ma->array_fields.size(); i++) {
    smt_astt tmp_idx = ctx->mk_smt_bvint(BigInt(i), false, dom_width);
    smt_astt idx_eq = real_idx->eq(ctx, tmp_idx);
    smt_astt val_eq = fresh->eq(ctx, ma->array_fields[i]);

    ctx->assert_ast(ctx->mk_func_app(bool_sort, SMT_FUNC_IMPLIES,
                                     idx_eq, val_eq));
  }

  return fresh;
}

smt_astt
array_convt::mk_store(const array_ast* ma, const expr2tc &idx,
                                smt_astt value, smt_sortt ressort)
{

  // Create a store: initially, consider whether to hand off to the unbounded
  // implementation.
  if (is_unbounded_array(ma->sort))
    return mk_unbounded_store(ma, idx, value, ressort);

  assert(ma->array_fields.size() != 0);

  array_ast *mast = new_ast(ressort, ma->array_fields);

  // If this is a constant index, simply update that particular field.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.value.to_ulong();
    if (intval > ma->array_fields.size())
      return ma;

    // Otherwise,
    mast->array_fields[intval] = value;
    return mast;
  }

  // For undetermined indexes, conditionally update each element of the bounded
  // array.
  smt_astt real_idx = ctx->convert_ast(idx);
  smt_astt real_value = value;
  unsigned long dom_width = mast->sort->domain_width;

  for (unsigned long i = 0; i < mast->array_fields.size(); i++) {
    smt_astt this_idx = ctx->mk_smt_bvint(BigInt(i), false, dom_width);
    smt_astt idx_eq = real_idx->eq(ctx, this_idx);

    smt_astt new_val = real_value->ite(ctx, idx_eq, mast->array_fields[i]);
    mast->array_fields[i] = new_val;
  }

  return mast;
}

smt_astt
array_convt::mk_unbounded_select(const array_ast *ma,
                                   const expr2tc &real_idx,
                                   smt_sortt ressort)
{
  // Store everything about this select, and return a free variable, that then
  // gets constrained at the end of conversion to tie up with the correct
  // value.

  // Record that we've accessed this index.
  idx_record new_idx_rec = { real_idx, ctx->ctx_level };
  array_indexes[ma->base_array_id].insert(new_idx_rec);

  // Corner case: if the idx we're selecting is the last one updated, just
  // use that piece of AST. This simplifies things later.
  const array_with &w = get_array_update(ma->base_array_id,
                                         ma->array_update_num);
  if (ma->array_update_num != 0 && !w.is_ite){
    if (real_idx == w.idx)
      return w.u.w.val;
  }

  // If the index has /already/ been selected for this particular array ast,
  // then we should return the fresh variable representing that select,
  // rather than adding another one.
  // XXX: this is a list/vec. Bad.
  array_select_containert::nth_index<0>::type &array_num_idx =
    array_selects[ma->base_array_id].get<0>();
  auto pair = array_num_idx.equal_range(ma->array_update_num);

  for (auto it = pair.first; it != pair.second; it++) {
    if (it->idx == real_idx) {
      // Aha.
      return it->val;
    }
  }

  // Generate a new free variable
  smt_astt a = ctx->mk_fresh(ressort, "mk_unbounded_select");

  struct array_select sel;
  sel.src_array_update_num = ma->array_update_num;
  sel.idx = real_idx;
  sel.val = a;
  sel.ctx_level = ctx->ctx_level;
  // Record this index
  array_selects[ma->base_array_id].insert(sel);

  // Convert index; it might trigger an array_of, or something else, which
  // fiddles with other arrays.
  ctx->convert_ast(real_idx);

  return a;
}

smt_astt
array_convt::mk_unbounded_store(const array_ast *ma,
                                  const expr2tc &idx, smt_astt value,
                                  smt_sortt ressort)
{
  // Store everything about this store, and suitably adjust all fields in the
  // array at the end of conversion so that they're all consistent.

  // Record that we've accessed this index.
  idx_record new_idx_rec = { idx, ctx->ctx_level };
  array_indexes[ma->base_array_id].insert(new_idx_rec);

  // More nuanced: allocate a new array representation.
  array_ast *newarr = new_ast(ressort);
  newarr->base_array_id = ma->base_array_id;
  newarr->array_update_num = array_updates[ma->base_array_id].size();

  // Record update
  struct array_with w;
  w.is_ite = false;
  w.idx = idx;
  w.u.w.src_array_update_num = ma->array_update_num;
  w.u.w.val = value;
  w.ctx_level = ctx->ctx_level;
  w.update_level = newarr->array_update_num;
  array_updates[ma->base_array_id].insert(w);

  // Add storage for the eventual collation of all these values
  array_valuation[ma->base_array_id].push_back(ast_vect());

  // Convert index; it might trigger an array_of, or something else, which
  // fiddles with other arrays.
  ctx->convert_ast(idx);

  // Result is the new array id goo.
  return newarr;
}

smt_astt
array_convt::array_ite(smt_astt cond,
                         const array_ast *true_arr,
                         const array_ast *false_arr,
                         smt_sortt thesort)
{

  // As ever, switch between ite's of unbounded arrays or bounded ones.
  if (is_unbounded_array(true_arr->sort))
    return unbounded_array_ite(cond, true_arr, false_arr, thesort);

  // For each element, make an ite.
  assert(true_arr->array_fields.size() != 0 &&
         true_arr->array_fields.size() == false_arr->array_fields.size());
  array_ast *mast = new_ast(thesort);
  unsigned long i;
  for (i = 0; i < true_arr->array_fields.size(); i++) {
    // One ite pls.
    smt_astt res = true_arr->array_fields[i]->ite(ctx, cond,
                                                  false_arr->array_fields[i]);
    mast->array_fields.push_back(array_downcast(res));
  }

  return mast;
}

smt_astt
array_convt::unbounded_array_ite(smt_astt cond,
                                   const array_ast *true_arr,
                                   const array_ast *false_arr,
                                   smt_sortt thesort)
{
  // We can perform ite's between distinct array id's, however the precondition
  // is that they must share the same set of array indexes, otherwise there's
  // the potential for data loss.

  unsigned int new_arr_id =
    std::min(true_arr->base_array_id, false_arr->base_array_id); // yolo

  array_ast *newarr = new_ast(thesort);
  newarr->base_array_id = new_arr_id;
  newarr->array_update_num = array_updates[true_arr->base_array_id].size();

  struct array_with w;
  w.is_ite = true;
  w.idx = expr2tc();
  w.u.i.true_arr_ast = true_arr;
  w.u.i.false_arr_ast = false_arr;
  w.u.i.cond = cond;
  w.ctx_level = ctx->ctx_level;
  w.update_level = newarr->array_update_num;
  array_updates[new_arr_id].insert(w);

  // Add storage for the eventual collation of all these values
  array_valuation[new_arr_id].push_back(ast_vect());

  return newarr;
}

smt_astt
array_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  // Create a new array, initialized with init_val
  smt_sortt dom_sort = ctx->mk_int_bv_sort(domain_width);
  smt_sortt idx_sort = init_val->sort;

  smt_sortt arr_sort = ctx->mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort);
  return convert_array_of_wsort(init_val, domain_width, arr_sort);
}

smt_astt
array_convt::convert_array_of_wsort(smt_astt init_val,
    unsigned long domain_width, smt_sortt arr_sort)
{
  smt_sortt idx_sort = init_val->sort;
  array_ast *mast = new_ast(arr_sort);

  if (is_unbounded_array(arr_sort)) {
    // If this is an unbounded array, simply store the value of the initializer
    // and constraint values at a later date. Heavy lifting is performed by
    // mk_array_symbol.
    std::string name = ctx->mk_fresh_name("array_of_unbounded::");
    mast = static_cast<array_ast*>(mk_array_symbol(name, arr_sort, idx_sort));

    struct array_of_val_rec v;
    v.array_id = mast->base_array_id;
    v.value = init_val;
    v.ctx_level = ctx->ctx_level;

    array_of_vals.insert(v);
  } else {
    // For bounded arrays, simply store the initializer in the explicit vector
    // of elements, x times.
    unsigned long array_size = 1UL << domain_width;
    for (unsigned long i = 0; i < array_size; i++)
      mast->array_fields.push_back(init_val);
  }

  return mast;
}

smt_astt
array_convt::encode_array_equality(const array_ast *a1, const array_ast *a2)
{
  // Record an equality between two arrays at this point in time. To be
  // implemented at constraint time.

  struct array_equality e;
  e.arr1_id = a1->base_array_id;
  e.arr2_id = a2->base_array_id;
  e.arr1_update_num = a1->array_update_num;
  e.arr2_update_num = a2->array_update_num;

  e.result = ctx->mk_fresh(ctx->boolean_sort, "");

  array_equalities.insert(std::make_pair(ctx->ctx_level, e));
  return e.result;
}

smt_astt
array_convt::mk_bounded_array_equality(const array_ast *a1, const array_ast *a2)
{
  assert(a1->array_fields.size() == a2->array_fields.size());

  smt_convt::ast_vec eqs;
  for (unsigned int i = 0; i < a1->array_fields.size(); i++) {
    eqs.push_back(a1->array_fields[i]->eq(ctx, a2->array_fields[i]));
  }

  return ctx->make_conjunct(eqs);
}

expr2tc
array_convt::get_array_elem(smt_astt a, uint64_t index, const type2tc &subtype)
{
  // During model building: get the value of an array at a particular, explicit,
  // index.
  const array_ast *mast = array_downcast(a);

  if (!is_unbounded_array(a->sort)) {
    if (index < mast->array_fields.size()) {
      return ctx->get_bv(subtype, mast->array_fields[index]);
    } else {
      return expr2tc(); // Out of range
    }
  }

  if (mast->base_array_id >= array_valuation.size()) {
    // This is an array that was not previously converted, therefore doesn't
    // appear in the valuation table. Therefore, all its values are free.
    return expr2tc();
  }

  // Fetch all the indexes
  const idx_record_containert &indexes = array_indexes[mast->base_array_id];
  unsigned int i = 0;

  // Basically, we have to do a linear search of all the indexes to find one
  // that matches the index argument.
  idx_record_containert::const_iterator it;
  for (it = indexes.begin(); it != indexes.end(); it++, i++) {
    const expr2tc &e = it->idx;
    expr2tc e2 = ctx->get(e);
    if (is_nil_expr(e2))
      continue;

    const constant_int2t &intval = to_constant_int2t(e2);
    if (intval.value.to_uint64() == index)
      break;
  }

  if (it == indexes.end())
    // Then this index wasn't modelled in any way.
    return expr2tc();

  // We've found an index; pick its value out, convert back to expr.

  const ast_vect &solver_values =
    array_valuation[mast->base_array_id][mast->array_update_num];
  assert(i < solver_values.size());

  if (array_subtypes[mast->base_array_id]->id == SMT_SORT_BOOL)
    return ctx->get_bool(solver_values[i]);
  else
    return ctx->get_bv(subtype, solver_values[i]);
}

void
array_convt::add_array_constraints_for_solving(void)
{

  join_array_indexes();
  add_new_indexes();
  execute_new_updates();
  apply_new_selects();
  add_array_equalities();

  return;
}

void
array_convt::push_array_ctx(void)
{
  // The most important factor in this process is to make sure that new indexes
  // in arrays are accounted for, as everything else is straightforwards. Thus,
  // the procedure is as follows:
  //  * Recompute array relations
  //  * Update indexes accordingly
  //  * Re-encode transitions for newly identified indexes
  //  * Apply array updates
  //  * Tie selects of historical array elements into unbounded selects
  //  * Apply new equalities and update old ones

  // Recomputes array relations, only for the current context level.
  join_array_indexes();

  // Allocate storage for new indexes, populate with variables, and add
  // historical constraints.
  add_new_indexes();

  // Obvious
  execute_new_updates();

  // Bind new selects into existing history in array valuations
  apply_new_selects();

  // Record how many arrays we had when this push occurred.
  num_arrays_history.push_back(array_valuation.size());
}

void
array_convt::pop_array_ctx(void)
{
  // Order of service:
  //  * Erase old array IDs
  //  * Erase old operations
  //  * Erase old indexes
  //  * Erase variable storage

  // CTX level will already have been decremented, we want to erase everything
  // to do with the previous context
  unsigned int target_ctx = ctx->ctx_level + 1;

  // Identify how many arrays we had at the last push, and reset back to that
  // number.
  unsigned int num_arrays = num_arrays_history[ctx->ctx_level];
  num_arrays_history.pop_back();

  // Demolish /all the things/
  array_subtypes.resize(num_arrays);
  array_selects.resize(num_arrays);
  array_updates.resize(num_arrays);
  array_indexes.resize(num_arrays);
  expr_index_map.resize(num_arrays);
  array_relations.resize(num_arrays);
  array_valuation.resize(num_arrays); // terrible terrible damage

  array_equalities.erase(target_ctx); // Erase everything with that idx
  auto &ctx_idx = array_of_vals.get<1>();
  ctx_idx.erase(target_ctx); // Similar

  // Now go through each storage Thing erasing any operations, indexes, or
  // whatever that were added in the old context level.
  for (auto &selects : array_selects) {
    auto &ctx_level_idx = selects.get<1>();
    ctx_level_idx.erase(target_ctx);
  }

  for (auto &updates : array_updates) {
    auto &ctx_level_idx = updates.get<1>();
    ctx_level_idx.erase(target_ctx);
  }

  for (auto &indexes : array_indexes) {
    auto &ctx_level_idx = indexes.get<1>();
    ctx_level_idx.erase(target_ctx);
  }

  for (auto &indexes : expr_index_map) {
    auto &ctx_level_idx = indexes.get<1>();
    ctx_level_idx.erase(target_ctx);
  }

  for (auto &relation : array_relations) {
    auto &ctx_level_idx = relation.get<1>();
    ctx_level_idx.erase(target_ctx);
  }

  // And now, in an intensely expensive operation, resize all the array value
  // vectors if they've had a change in number of indexes.
  for (unsigned int arrid = 0; arrid < array_updates.size(); arrid++) {
    unsigned int num_indexes = array_indexes[arrid].size();
    if (array_valuation[arrid][0].size() != num_indexes) {
      // Index size has changed. Resize /all the things/
      for (ast_vect &vec : array_valuation[arrid]) {
        vec.resize(num_indexes);
      }
    }
  }

  // Uh. Fini?
}

void
array_convt::join_array_indexes()
{
  // Identify the set of array ID's that, due to equalities and ITE's, are
  // effectively joined into the same array. For each of these sets, join their
  // indexes.
  // This needs to support transitivity.

  array_relations.resize(array_updates.size());

  // Load the existing set of id's into the groupings vector.

  // Collect together the set of array id's touched by each array id.
  unsigned int arrid = 0;
  for (unsigned int arrid = 0; arrid < array_updates.size(); arrid++) {
    touched_array_sett &joined_array_ids = array_relations[arrid];

    for (const auto &update : array_updates[arrid]) {
      if (update.is_ite) {
        if (update.u.i.true_arr_ast->base_array_id !=
            update.u.i.false_arr_ast->base_array_id) {
          touched_arrayt t;

          t.array_id = update.u.i.true_arr_ast->base_array_id;
          t.ctx_level = ctx->ctx_level;
          joined_array_ids.insert(t);

          t.array_id = update.u.i.false_arr_ast->base_array_id;
          t.ctx_level = ctx->ctx_level;
          joined_array_ids.insert(t);
        }
      }
    }
  }

  for (const auto &equality : array_equalities) {
    touched_arrayt t;

    t.array_id = equality.second.arr2_id;
    t.ctx_level = ctx->ctx_level;
    array_relations[equality.second.arr1_id].insert(t);

    t.array_id = equality.second.arr1_id;
    t.ctx_level = ctx->ctx_level;
    array_relations[equality.second.arr2_id].insert(t);
  }

  // K; now compute a fixedpoint joining the sets of things that touch each
  // other.
  bool modified = false;
  do {
    modified = false;

    for (const auto &arrset : array_relations) {
      for (const auto &touched_arr_rec : arrset) {
        // It the other array recorded as touching all the arrays that this one
        // does? Try inserting this set, and see if the size changes. Slightly
        // ghetto, but avoids additional allocations.
        unsigned int original_size =
          array_relations[touched_arr_rec.array_id].size();

        // Attempt insertions, modfiying context level to the current one.
        // This is necessary because this relation needs to not exist after a
        // pop of this operation.
        for (const auto &other_arr_rec : arrset) {
          touched_arrayt t = other_arr_rec;
          t.ctx_level = ctx->ctx_level;

          // As array_id index is unique, this will be rejected if it's already
          // recorded, even if the context level is different.
          array_relations[touched_arr_rec.array_id].insert(t);
        }

        if (original_size != array_relations[touched_arr_rec.array_id].size())
          modified = true;
      }
    }
  } while (modified);

  // Right -- now join all ther indexes. This can be optimised, but not now.
  for (arrid = 0; arrid < array_updates.size(); arrid++) {
    const auto &arrset = array_relations[arrid];
    for (const auto &touched_arr_rec : arrset) {
      // Only juggle indexes for relations that were just encoded
      if (touched_arr_rec.ctx_level != ctx->ctx_level)
        continue;

      // Go through each index of the other array and bake it into the current
      // selected one if it isn't already.
      for (const auto &idx : array_indexes[touched_arr_rec.array_id]) {
        if (array_indexes[arrid].find(idx.idx) == array_indexes[arrid].end()) {
          idx_record r;
          r.idx = idx.idx;
          r.ctx_level = ctx->ctx_level;
          array_indexes[arrid].insert(r);
        }
      }
    }
  }

  // Le fin
  return;
}

void
array_convt::add_new_indexes()
{
  // In the context of a push/pop, we may have collected some new indexes in an
  // array as the result of, well, any array operation. These make their way
  // into the array_indexes store in the following ways:
  //  * Selects: inserted directly by mk_unbounded_select
  //  * Updates: like selects
  //  * ITE's: for same-array-id ITE's, same as updates. However for cross array
  //    id's, join_array_indexes must be called again to re-collate array
  //    relations and join array indexes
  //  * Equalities: like cross-array ITE's
  //
  // So, fetch a list of any new array indexes, and perform the relevant
  // computations for them. We need to apply all the historical operations:
  // ackerman constraints, ite's and updates, as the index might be the same
  // concrete index as another symbolic one.
  //
  // Happily we can just juggle some storage, then use existing functions to
  // operate only on the subset of new indexes.

  // Vector with flags indicating whether we need to re-execute
  std::vector<bool> re_execute;
  std::vector<unsigned int> start_pos;
  unsigned int arrid = 0;
  for (const idx_record_containert &rec : array_indexes) {
    auto &ctx_index = rec.get<1>();
    auto pair = ctx_index.equal_range(ctx->ctx_level);

    if (pair.first == pair.second) {
      // Nothing new in this ctx level
      re_execute.push_back(false);
      start_pos.push_back(0);
      arrid++;
      continue;
    }

    re_execute.push_back(true);
    start_pos.push_back(expr_index_map[arrid].size());

    // We're guarenteed that each of these indexes are _new_ to this array.
    // Enumerate them, giving them a location in the expr_index_map.
    index_map_containert &idx_map = expr_index_map[arrid];
    for (auto it = pair.first; it != pair.second; it++) {
      struct index_map_rec r;
      r.idx = it->idx;
      r.vec_idx = idx_map.size();
      r.ctx_level = ctx->ctx_level;
      idx_map.insert(r);
    }

    arrid++;
  }

  // Now that we've allocated vector index locations, resize the array valuation
  // vector(s) to have storage for that many ast members.
  arrid = 0;
  for (arrid = 0; arrid < array_valuation.size(); arrid++ ) {
    array_update_vect &valuation = array_valuation[arrid];
    unsigned int num_indexes = array_indexes[arrid].size();

    for (ast_vect &vec : valuation) {
        assert(vec.size() <= num_indexes &&
               "Array valuations should only ever increase in size in a push");
      vec.resize(num_indexes);
    }
  }

  // And now: perform the relevant transitions to populate those new indexes
  // with valuations. Start by populating the initial set of values and applying
  // the initial ackerman constraints to them.

  for (arrid = 0; arrid < array_updates.size(); arrid++) {
    if (!re_execute[arrid])
      continue;

    array_update_vect &array_values = array_valuation[arrid];
    smt_sortt subtype = array_subtypes[arrid];

    // Fill inital values with either free variables or the initialiser
    array_of_val_containert::nth_index<0>::type &array_num_idx =
      array_of_vals.get<0>();
    auto it = array_num_idx.find(arrid);

    if (it != array_num_idx.end()) {
      collate_array_values(array_values[0], arrid, 0, subtype, start_pos[arrid],
          it->value);
    } else {
      collate_array_values(array_values[0], arrid, 0, subtype,
          start_pos[arrid]);
    }

    // Apply inital ackerman constraints
    add_initial_ackerman_constraints(array_values[0], expr_index_map[arrid],
        start_pos[arrid]);

    // And finally, re-execute the relevant array transitions
    for (unsigned int i = 0; i < array_updates[arrid].size() - 1; i++)
      execute_array_trans(array_values, arrid, i, subtype, start_pos[arrid]);
  }
}

void
array_convt::execute_new_updates(void)
{
  // Identify new array updates, and execute them.

  for (unsigned int arrid = 0; arrid < array_updates.size(); arrid++) {
    smt_sortt subtype = array_subtypes[arrid];
    array_update_containert &updates = array_updates[arrid];
    auto &update_index = updates.get<0>();

    // We need to execute the updates in order. So, use the array update index,
    // walk backwards to the point where the updates in this context start,
    // storing pointers to any updates we find along the way with this context
    // level.

    std::list<const array_witht *> withs;
    auto rit = update_index.rbegin();
    while (rit != update_index.rend()) {
      if (rit->ctx_level == ctx->ctx_level)
        withs.push_back(&(*rit));
      else if (rit->ctx_level != UINT_MAX) // ahem
        break;
      rit++;
    }

    if (withs.size() == 0)
      continue;

    // We've identified where to start encoding transitions -- from rit back
    // to the beginning. Go backwards through the iterator, encoding them.
    for (auto ptr : withs) {
      execute_array_trans(array_valuation[arrid], arrid, ptr->update_level - 1,
          subtype, 0);
    }
  }
}

void
array_convt::apply_new_selects(void)
{
  // In the push context procedure, two kinds of new selects have already been
  // encoded. They're ones that either apply to a new index expr (through the
  // use of collate_array_values), and those that apply to new update indexes
  // (through execute_new_updates).
  // That then leaves new selects that apply to previously encoded array
  // values. We can just pick those straight out of the array valuation vector.
  // This could be optimised, but not now.

  for (unsigned int arrid = 0; arrid < array_selects.size(); arrid++) {
    array_select_containert &selects = array_selects[arrid];
    // Look up selects by context level
    auto &ctx_level_index = selects.get<1>();
    auto pair = ctx_level_index.equal_range(ctx->ctx_level);

    // No selects?
    if (pair.first == pair.second)
      continue;

    // Go through each of these selects.
    for (auto it = pair.first; it != pair.second; it++) {
      // Look up where in the valuation this is.
      auto index_rec = expr_index_map[arrid].find(it->idx);
      smt_astt &dest =
        array_valuation[arrid][it->src_array_update_num][index_rec->vec_idx];

      // OK. We can know that one of the two already-done cases described above
      // have happened, if the current AST pointer is the one from this select.
      // If it isn't one of those cases, it will have been filled with a free
      // value, then constrained as appropriate
      if (dest == it->val)
        continue;

      // OK, bind this select in through an equality.
      ctx->assert_ast(dest->eq(ctx, it->val));
    }
  }
}

void
array_convt::add_array_equalities(void)
{
  // Precondition: all constraints have already been added and constrained into
  // the array_valuation vectors. Also that the array ID's being used all share
  // the same indexes.

  // Pick only equalities that have been encoded in the current ctx.
  auto pair = array_equalities.equal_range(ctx->ctx_level);

  for (auto it = pair.first; it != pair.second; it++) {
    assert(array_indexes_are_same(array_indexes[it->second.arr1_id],
                                  array_indexes[it->second.arr2_id]));

    add_array_equality(it->second.arr1_id, it->second.arr2_id,
                       it->second.arr1_update_num, it->second.arr2_update_num,
                       it->second.result);
  }

  // Second phase: look at all past equalities to see whether or not they need
  // to be extended to account for new indexes.
  for (auto it = array_equalities.begin(); it != array_equalities.end(); it++) {
    // Don't touch equalities we've already done
    if (it->first == ctx->ctx_level)
      continue;

    idx_record_containert &idxs = array_indexes[it->second.arr1_id];
    idx_record_containert::nth_index<1>::type &ctx_level_idx = idxs.get<1>();
    auto pair = ctx_level_idx.equal_range(ctx->ctx_level);

    if (pair.first == pair.second)
      // No indexes added in this context level, no additional constraints
      // required
      continue;

    // Ugh. We need to know how many new indexes there are, and where to start
    // in the array valuation array. So, count them.
    unsigned int ctx_count = 0;
    for (auto it2 = pair.first; it2 != pair.second; it2++)
      ctx_count++;

    // All the new indexes will have been appended to the vector, so the start
    // pos is the number of elements, minus the ones on the end.
    unsigned int start_pos =
      array_indexes[it->second.arr1_id].size() - ctx_count;

    // There are new indexes; apply equalities.
    add_array_equality(it->second.arr1_id, it->second.arr2_id,
        it->second.arr1_update_num, it->second.arr2_update_num,
        it->second.result, start_pos);
  }
}

void
array_convt::add_array_equality(unsigned int arr1_id, unsigned int arr2_id,
                                unsigned int arr1_update,
                                unsigned int arr2_update,
                                smt_astt result,
                                unsigned int start_pos)
{
  // Simply get a handle on two vectors of valuations in array_valuation,
  // and encode an equality.
  const ast_vect &a1 = array_valuation[arr1_id][arr1_update];
  const ast_vect &a2 = array_valuation[arr2_id][arr2_update];

  smt_convt::ast_vec lits;
  assert(start_pos < a1.size());
  for (unsigned int i = start_pos; i < a1.size(); i++) {
    lits.push_back(a1[i]->eq(ctx, a2[i]));
  }

  smt_astt conj = ctx->make_conjunct(lits);
  ctx->assert_ast(result->eq(ctx, conj));
  return;
}

void
array_convt::execute_array_trans(array_update_vect &data,
    unsigned int arr, unsigned int idx, smt_sortt subtype,
    unsigned int start_point)
{
  // Encode the constraints for a particular array update.

  // Steps: First, fill the destination vector with either free variables, or
  // the free variables from selects corresponding to that item.
  // Then apply update or ITE constraints.
  // Then apply equalities between the old and new values.

  // The destination vector: representing the values of each element in the
  // next updated state.
  assert(idx+1 < data.size());
  ast_vect &dest_data = data[idx+1];

  // Fill dest_data with ASTs: if a select has been applied for a particular
  // index, then that value is inserted there. Otherwise, a free value is
  // inserted.
  collate_array_values(dest_data, arr, idx+1, subtype, start_point);

  // Two updates that could have occurred for this array: a simple with, or
  // an ite.
  assert(idx+1 < array_updates[arr].size());
  const array_with &w = get_array_update(arr, idx+1);
  if (w.is_ite) {
    if (w.u.i.true_arr_ast->base_array_id !=
        w.u.i.false_arr_ast->base_array_id) {
      execute_array_joining_ite(dest_data, arr, w.u.i.true_arr_ast,
                                w.u.i.false_arr_ast,
                                expr_index_map[arr],
                                w.u.i.cond,
                                subtype,
                                start_point);
    } else {
      unsigned int true_idx = w.u.i.true_arr_ast->array_update_num;
      unsigned int false_idx = w.u.i.false_arr_ast->array_update_num;
      assert(true_idx < idx + 1 && false_idx < idx + 1);
      execute_array_ite(dest_data, data[true_idx], data[false_idx],
                        expr_index_map[arr], w.u.i.cond, start_point);
    }
  } else {
    execute_array_update(dest_data, data[w.u.w.src_array_update_num],
                         expr_index_map[arr], w.idx, w.u.w.val, start_point);
  }
}

void
array_convt::execute_array_update(ast_vect &dest_data,
  ast_vect &source_data,
  const index_map_containert &idx_map,
  const expr2tc &idx,
  smt_astt updated_value,
  unsigned int start_point)
{
  // Place a constraint on the updated variable; add equality constraints
  // between the older version and this version.

  // So, the updated element,
  auto it = idx_map.find(idx);
  assert(it != idx_map.end());

  smt_astt update_idx_ast = ctx->convert_ast(idx);
  unsigned int updated_idx = it->vec_idx;

  // Assign in its value. Note that no selects occur agains this data index,
  // they will have been replaced with the update ast when the select was
  // encoded.
  dest_data[updated_idx] = updated_value;

  for (auto it2 = idx_map.begin(); it2 != idx_map.end(); it2++) {
    if (it2->vec_idx == updated_idx)
      continue;

    if (it2->vec_idx < start_point)
      continue;

    // Generate an ITE. If the index is nondeterministically equal to the
    // current index, take the updated value, otherwise the original value.
    // This departs from the CBMC implementation, in that they explicitly
    // use implies and ackerman constraints.
    // FIXME: benchmark the two approaches. For now, this is shorter.
    smt_astt cond = update_idx_ast->eq(ctx, ctx->convert_ast(it2->idx));
    smt_astt dest_ite = updated_value->ite(ctx, cond, source_data[it2->vec_idx]);
    ctx->assert_ast(dest_data[it2->vec_idx]->eq(ctx, dest_ite));
  }

  return;
}

void
array_convt::execute_array_ite(ast_vect &dest,
    const ast_vect &true_vals,
    const ast_vect &false_vals,
    const index_map_containert &idx_map,
    smt_astt cond,
    unsigned int start_point)
{

  // Each index value becomes an ITE between each source value.
  for (unsigned int i = start_point; i < idx_map.size(); i++) {
    smt_astt updated_elem = true_vals[i]->ite(ctx, cond, false_vals[i]);
    ctx->assert_ast(dest[i]->eq(ctx, updated_elem));
  }

  return;
}

void
array_convt::execute_array_joining_ite(ast_vect &dest,
    unsigned int cur_id, const array_ast *true_arr_ast,
    const array_ast *false_arr_ast, const index_map_containert &idx_map,
    smt_astt cond, smt_sortt subtype, unsigned int start_point)
{

  const array_ast *local_ast, *remote_ast;
  bool local_arr_values_are_true = (true_arr_ast->base_array_id == cur_id);
  if (local_arr_values_are_true) {
    local_ast = true_arr_ast;
    remote_ast = false_arr_ast;
  } else {
    local_ast = false_arr_ast;
    remote_ast = true_arr_ast;
  }

  ast_vect selects;
  selects.reserve(array_indexes[cur_id].size());
  assert(array_indexes_are_same(array_indexes[cur_id],
                                array_indexes[remote_ast->base_array_id]));

  for (const auto &elem : array_indexes[remote_ast->base_array_id]) {
    selects.push_back(mk_unbounded_select(remote_ast, elem.idx, subtype));
  }

  // Now select which values are true or false
  const ast_vect *true_vals, *false_vals;
  if (local_arr_values_are_true) {
    true_vals =
      &array_valuation[local_ast->base_array_id][local_ast->array_update_num];
    false_vals = &selects;
  } else {
    false_vals =
      &array_valuation[local_ast->base_array_id][local_ast->array_update_num];
    true_vals = &selects;
  }

  execute_array_ite(dest, *true_vals, *false_vals, idx_map, cond,
      start_point);

  return;
}

void
array_convt::collate_array_values(ast_vect &vals,
                                    unsigned int base_array_id,
                                    unsigned int array_update_num,
                                    smt_sortt subtype,
                                    unsigned int start_point,
                                    smt_astt init_val)
{
  // IIRC, this translates the history of an array + any selects applied to it,
  // into a vector mapping a particular index to the variable representing the
  // element at that index. XXX more docs.

  const index_map_containert &idx_map = expr_index_map[base_array_id];

  // So, the value vector should be allocated but not initialized,
  assert(vals.size() == idx_map.size());

  // First, make everything null,
  for (unsigned int i = start_point; i < vals.size(); i++)
    vals[i] = NULL;

  // Get the range of values with this update array num.
  array_select_containert &idxs = array_selects[base_array_id];
  array_select_containert::nth_index<0>::type &array_num_idx = idxs.get<0>();
  auto pair = array_num_idx.equal_range(array_update_num);

  // Now assign in all free variables created as a result of selects.
  for (auto it = pair.first; it != pair.second; it++) {
    auto it2 = idx_map.find(it->idx);
    assert(it2 != idx_map.end());

    if (it2->vec_idx < start_point)
      continue;

    vals[it2->vec_idx] = it->val;
  }

  // Initialize everything else to either a free variable or the initial value.
  if (init_val == NULL) {
    // Free variables, except where free variables tied to selects have occurred
    for (unsigned int vec_idx = 0; vec_idx < vals.size(); vec_idx++) {
      if (vals[vec_idx] == NULL)
        vals[vec_idx] = ctx->mk_fresh(subtype, "collate_array_vals::");
    }
  } else {
    // We need to assign the initial value in, except where there's already
    // a select/index, in which case we assert that the values are equal.
    for (auto it = vals.begin(); it != vals.end(); it++) {
      if (*it == NULL) {
        *it = init_val;
      } else {
        ctx->assert_ast((*it)->eq(ctx, init_val));
      }
    }
  }

  // Fin.
}

void
array_convt::add_initial_ackerman_constraints(
                                  const ast_vect &vals,
                                  const index_map_containert &idx_map,
                                  unsigned int start_point)
{
  // Add ackerman constraints: these state that for each element of an array,
  // where the indexes are equivalent (in the solver), then the value of the
  // elements are equivalent. The cost is quadratic, alas.

  smt_sortt boolsort = ctx->boolean_sort;
  for (auto it = idx_map.begin(); it != idx_map.end(); it++) {
     if (it->vec_idx < start_point)
       continue;

    smt_astt outer_idx = ctx->convert_ast(it->idx);
    for (auto it2 = idx_map.begin(); it2 != idx_map.end(); it2++) {
      smt_astt inner_idx = ctx->convert_ast(it2->idx);

      // If they're the same idx, they're the same value.
      smt_astt idxeq = outer_idx->eq(ctx, inner_idx);

      smt_astt valeq = vals[it->vec_idx]->eq(ctx, vals[it2->vec_idx]);

      ctx->assert_ast(ctx->mk_func_app(boolsort, SMT_FUNC_IMPLIES,
                                       idxeq, valeq));
    }
  }
}

smt_astt
array_ast::eq(smt_convt *ctx __attribute__((unused)), smt_astt sym) const
{
  const array_ast *other = array_downcast(sym);

  if (is_unbounded_array(sort)) {
    return array_ctx->encode_array_equality(this, other);
  } else {
    return array_ctx->mk_bounded_array_equality(this, other);
  }
}

void
array_ast::assign(smt_convt *ctx __attribute__((unused)), smt_astt sym) const
{
  array_ctx->convert_array_assign(this, sym);
}

smt_astt
array_ast::update(smt_convt *ctx __attribute__((unused)), smt_astt value,
                                unsigned int idx,
                                expr2tc idx_expr) const
{
  if (is_nil_expr(idx_expr))
    idx_expr = constant_int2tc(get_uint_type(sort->domain_width), BigInt(idx));

  return array_ctx->mk_store(this, idx_expr, value, sort);
}

smt_astt
array_ast::select(smt_convt *ctx __attribute__((unused)),
                  const expr2tc &idx) const
{
  // Look up the array subtype sort. If we're unbounded, use the base array id
  // to do that, otherwise pull the subtype out of an element.
  smt_sortt s;
  if (!array_fields.empty())
    s = array_fields[0]->sort;
  else
    s = array_ctx->array_subtypes[base_array_id];

  return array_ctx->mk_select(this, idx, s);
}

smt_astt
array_ast::ite(smt_convt *ctx __attribute__((unused)),
               smt_astt cond, smt_astt falseop) const
{

  return array_ctx->array_ite(cond, this, array_downcast(falseop), sort);
}
