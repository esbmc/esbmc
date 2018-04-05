#ifndef _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_
#define _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_

// Something to abstract the flattening of arrays to QF_BV.

// jmorse: this originally began it's life as a flatten-arrays-to-bitvectors
// implementation, largely derived from the CBMC implementation, with some
// adjustments. It then, however, became necessary for struct arrays, because
// while we can work around structs not being available in SMT, we need this
// kind of array implementation to handle arrays of arbitary types.
//
// Even more complexity turns up with the "smt-during-symex" facility, in that
// one must be able to perform a series of array operations and flush them into
// the SMT solver in a way that is compatible with pushing/popping of
// constraints.
//
// As a result, this particular class is due some serious maintenence.

#include <set>
#include <solvers/smt/smt_conv.h>
#include <util/irep2.h>

static inline bool is_unbounded_array(const smt_sort *s)
{
  if(s->id != SMT_SORT_ARRAY)
    return false;

  // This is either really large, or unbounded thus leading to a machine_int
  // sized domain. Either way, not a normal one.
  if(s->get_domain_width() > 10)
    return true;

  return false;
}

class array_convt;

class array_ast : public smt_ast
{
public:
#define array_downcast(x) static_cast<const array_ast *>(x)

  array_ast(array_convt *actx, smt_convt *ctx, const smt_sort *_s)
    : smt_ast(ctx, _s), symname(""), array_ctx(actx)
  {
  }

  array_ast(
    array_convt *actx,
    smt_convt *ctx,
    const smt_sort *_s,
    std::vector<smt_astt> _a)
    : smt_ast(ctx, _s),
      symname(""),
      array_fields(std::move(_a)),
      array_ctx(actx)
  {
  }

  ~array_ast() override = default;

  smt_astt eq(smt_convt *ctx, smt_astt other) const override;
  smt_astt ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const override;
  void assign(smt_convt *ctx, smt_astt sym) const override;
  smt_astt update(
    smt_convt *ctx,
    smt_astt value,
    unsigned int idx,
    expr2tc idx_expr = expr2tc()) const override;
  smt_astt select(smt_convt *ctx, const expr2tc &idx) const override;

  smt_astt eq_fixedsize(smt_convt *ctx, const array_ast *other) const;

  void dump() const override
  {
    std::cout << "name: " << symname << '\n';
    for(auto const &e : array_fields)
      e->dump();
  }

  std::string symname; // Only if this was produced from mk_smt_symbol.

  std::vector<smt_astt> array_fields;
  unsigned int base_array_id;
  unsigned int array_update_num;

  array_convt *array_ctx;
};

class array_convt : public array_iface
{
public:
  struct array_select;
  struct array_with;
  typedef smt_convt::ast_vec ast_vect;
  typedef std::vector<ast_vect> array_update_vect;

  // Fgasdf
  struct index_map_rec
  {
    expr2tc idx;
    unsigned int vec_idx;
    unsigned int ctx_level;
  };
  typedef struct index_map_rec index_map_rect;

  typedef boost::multi_index_container<
    index_map_rect,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(index_map_rect, expr2tc, idx)>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(index_map_rect, unsigned int, ctx_level),
        std::greater<unsigned int>>>>
    index_map_containert;

  array_convt(smt_convt *_ctx);
  ~array_convt() = default;

  // Public api
  smt_ast *mk_array_symbol(
    const std::string &name,
    const smt_sort *ms,
    smt_sortt subtype) override;
  expr2tc
  get_array_elem(smt_astt a, uint64_t index, const type2tc &subtype) override;
  smt_astt
  convert_array_of(smt_astt init_val, unsigned long domain_width) override;
  void add_array_constraints_for_solving() override;

  // Heavy lifters
  virtual smt_astt convert_array_of_wsort(
    smt_astt init_val,
    unsigned long domain_width,
    smt_sortt arr_sort);
  unsigned int new_array_id();
  void convert_array_assign(const array_ast *src, smt_astt sym);
  smt_astt mk_select(
    const array_ast *array,
    const expr2tc &idx,
    const smt_sort *ressort);
  virtual smt_astt mk_store(
    const array_ast *array,
    const expr2tc &idx,
    smt_astt value,
    const smt_sort *ressort);
  smt_astt mk_bounded_array_equality(const array_ast *a1, const array_ast *a2);

  smt_astt array_ite(
    smt_astt cond,
    const array_ast *true_arr,
    const array_ast *false_arr,
    const smt_sort *thesort);

  // Internal funk:

  smt_astt encode_array_equality(const array_ast *a1, const array_ast *a2);
  smt_astt mk_unbounded_select(
    const array_ast *array,
    const expr2tc &idx,
    const smt_sort *ressort);
  smt_astt mk_unbounded_store(
    const array_ast *array,
    const expr2tc &idx,
    smt_astt value,
    const smt_sort *ressort);
  smt_astt unbounded_array_ite(
    smt_astt cond,
    const array_ast *true_arr,
    const array_ast *false_arr,
    const smt_sort *thesort);

  // Array constraint beating

  void join_array_indexes();
  void add_array_equalities();
  void add_array_equality(
    unsigned int arr1_id,
    unsigned int arr2_id,
    unsigned int arr1_update,
    unsigned int arr2_update,
    smt_astt &result,
    unsigned int start_pos = 0);
  void execute_array_trans(
    array_update_vect &data,
    unsigned int arr,
    unsigned int idx,
    const smt_sort *subtype,
    unsigned int start_point);
  void execute_array_update(
    ast_vect &dest_data,
    ast_vect &src_data,
    const index_map_containert &idx_map,
    const expr2tc &idx,
    smt_astt val,
    unsigned int start_point);
  void execute_array_ite(
    ast_vect &dest,
    const ast_vect &true_vals,
    const ast_vect &false_vals,
    const index_map_containert &idx_map,
    smt_astt cond,
    unsigned int start_point);
  void execute_array_joining_ite(
    ast_vect &dest,
    unsigned int cur_id,
    const array_ast *true_arr_ast,
    const array_ast *false_arr_ast,
    const index_map_containert &idx_map,
    smt_astt cond,
    smt_sortt subtype,
    unsigned int start_point);

  void collate_array_values(
    ast_vect &vals,
    unsigned int base_array_id,
    unsigned int array_update_no,
    const smt_sort *subtype,
    unsigned int start_point,
    smt_astt init_val = nullptr);
  void add_initial_ackerman_constraints(
    const ast_vect &vals,
    const index_map_containert &idx_map,
    unsigned int start_point);
  void add_new_indexes();
  void execute_new_updates();
  void apply_new_selects();

  inline array_ast *new_ast(smt_sortt _s)
  {
    return new array_ast(this, ctx, _s);
  }

  inline array_ast *new_ast(smt_sortt _s, const std::vector<smt_astt> &_a)
  {
    return new array_ast(this, ctx, _s, _a);
  }

  void push_array_ctx() override;
  void pop_array_ctx() override;

  // Members

  // Array tracking: each new root array (from fresh_array) gets its own
  // ID number which is stored. Then, whenever any operation occurs on it,
  // we add the index to the set contained in the following object. This
  // obtains all the tracking data required for CBMC-like array
  // bitblasting.
  // Various technical jiggery pokery occurs to scope these indexes.

  struct idx_record
  {
    expr2tc idx;
    unsigned int ctx_level;
  };

  typedef boost::multi_index_container<
    idx_record,
    boost::multi_index::indexed_by<
      boost::multi_index::hashed_unique<
        BOOST_MULTI_INDEX_MEMBER(idx_record, expr2tc, idx)>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(idx_record, unsigned int, ctx_level),
        std::greater<unsigned int>>>>
    idx_record_containert;
  std::vector<idx_record_containert> array_indexes;

  // Self explanatory. Contains bitwidth of subtypes
  std::vector<smt_sortt> array_subtypes;

  // Array /value/ tracking. For each array (outer vector) we have an inner
  // vector, each element of which corresponds to each 'with' operation
  // on an array. Within that is a list of indexes and free value'd
  // elements: whenever we select an element from an array, we return a
  // free value, and record it here. Assertions made later will link this
  // up with real values.
  struct array_select
  {
    unsigned int src_array_update_num;
    expr2tc idx;
    // Mutable because this might be used as a vehicle to read from
    // array_valuations
    mutable smt_astt val;
    unsigned int ctx_level;
    mutable bool converted;
  };
  typedef struct array_select array_selectt;

  typedef boost::multi_index_container<
    array_selectt,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(
          array_selectt,
          unsigned int,
          src_array_update_num),
        std::greater<unsigned int>>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(array_selectt, unsigned int, ctx_level),
        std::greater<unsigned int>>>>
    array_select_containert;

  std::vector<array_select_containert> array_selects;

  // Array equalities -- decomposed into selects when array constraints
  // are encoded.
  // Each vector element corresponds to an array, containing a list of
  // equalities. Each equality has the current update number, the details
  // of the other array, and the output ast.
  struct array_equality
  {
    unsigned int arr1_id;
    unsigned int arr2_id;

    unsigned int arr1_update_num;
    unsigned int arr2_update_num;

    // Mutable to allow the creation of an equality without a symbol, then
    // reading that from this structure.
    mutable smt_astt result;
  };

  std::multimap<unsigned int, struct array_equality> array_equalities;

  // Update records: For each array, for each 'with' operation, we record
  // the index used and the AST representation of the value assigned. We
  // also store the ID number of the source array, because due to phi's
  // we may branch arrays before joining.
  // We also record ite's here, because they're effectively an array update.
  // The is_ite boolean indicates whether the following union is a with or
  // an ite repr. If it's an ite, the two integers represent which two
  // historical update indexes of the array are operands of the ite.
  struct array_with
  {
    bool is_ite;
    expr2tc idx;
    union {
      struct
      {
        unsigned int src_array_update_num;
        smt_astt val;
      } w;
      struct
      {
        unsigned int src_array_id_true;
        unsigned int src_array_update_true;
        unsigned int src_array_id_false;
        unsigned int src_array_update_false;
        const array_ast *true_arr_ast;  // yolo
        const array_ast *false_arr_ast; // yolo
        smt_astt cond;
      } i;
    } u;
    unsigned int ctx_level;
    unsigned int update_level;
    mutable bool converted;
  };
  typedef struct array_with array_witht;

  typedef boost::multi_index_container<
    array_witht,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique<
        BOOST_MULTI_INDEX_MEMBER(array_witht, unsigned int, update_level),
        std::greater<unsigned int>>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(array_witht, unsigned int, ctx_level),
        std::greater<unsigned int>>>>
    array_update_containert;

  std::vector<array_update_containert> array_updates;

  inline const array_with &get_array_update(unsigned int id, unsigned int up)
  {
    array_update_containert::nth_index<0>::type &updated_idx =
      array_updates[id].get<0>();
    auto it = updated_idx.find(up);
    assert(it != updated_idx.end());
    return *it;
  }

  // Map between base array identifiers and the value to initialize it with.
  // Only applicable to unbounded arrays.
  struct array_of_val_rec
  {
    unsigned int array_id;
    smt_astt value;
    unsigned int ctx_level;
  };
  typedef struct array_of_val_rec array_of_val_rect;

  typedef boost::multi_index_container<
    array_of_val_rect,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique<
        BOOST_MULTI_INDEX_MEMBER(array_of_val_rect, unsigned int, array_id),
        std::greater<unsigned int>>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(array_of_val_rect, unsigned int, ctx_level),
        std::greater<unsigned int>>>>
    array_of_val_containert;

  array_of_val_containert array_of_vals;

  // Index map, between index expressions, and where the value for that index
  // actually sits in ast vectors.
  std::vector<index_map_containert> expr_index_map;

  // Storage of the sets of arrays that other arrays interact with.
  struct touched_array
  {
    unsigned int array_id;
    unsigned int ctx_level;
  };
  typedef struct touched_array touched_arrayt;

  typedef boost::multi_index_container<
    touched_arrayt,
    boost::multi_index::indexed_by<
      boost::multi_index::ordered_unique<
        BOOST_MULTI_INDEX_MEMBER(touched_arrayt, unsigned int, array_id),
        std::greater<unsigned int>>,
      boost::multi_index::ordered_non_unique<
        BOOST_MULTI_INDEX_MEMBER(touched_arrayt, unsigned int, ctx_level),
        std::greater<unsigned int>>>>
    touched_array_sett;
  std::vector<touched_array_sett> array_relations;

  // History of when different array IDs were allocated. For a change,
  // indexed by the context level.
  std::vector<unsigned int> num_arrays_history;

  // Finally, for model building, we need all the past array values. Three
  // vectors, dimensions are arrays id's, historical point, array element,
  // respectively.
  // In reverse, these correspond to ast_vect and array_update_vect
  std::vector<std::vector<std::vector<smt_astt>>> array_valuation;

  smt_convt *ctx;
};

#endif /* _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_ */
