#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/numpy/numpy_nditer.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>

#include <util/lang/c_types.h>

#include <cassert>
#include <algorithm>
#include <map>
#include <stdexcept>

// Rebinding a NumPy array name leaves the old array, and every view of it,
// untouched. Views of one storage are repointed together into one snapshot
// of that storage, so they keep aliasing each other.

void python_converter::share_numpy_storage_snapshot(
  const std::string &storage_id,
  const std::vector<std::string> &view_ids,
  const locationt &location,
  codet &target_block,
  const std::string &heir_id)
{
  const namespacet ns(symbol_table_);
  const symbolt *source = symbol_table_.find_symbol(storage_id);
  assert(source);
  const typet source_type = ns.follow(source->get_type());
  assert(
    source_type.is_array() && to_array_type(source_type).size().is_constant());

  typet scalar_type = source_type;
  while (scalar_type.is_array())
    scalar_type = ns.follow(to_array_type(scalar_type).subtype());
  const pointer_typet scalar_ptr(scalar_type);

  // An alias that inherits the storage already holds the copy.
  symbolt &snapshot =
    heir_id.empty()
      ? create_tmp_symbol(location, "$rebind_snapshot$", source_type, exprt())
      : *symbol_table_.find_symbol(heir_id);
  if (heir_id.empty())
  {
    code_declt snapshot_decl(symbol_expr(snapshot));
    snapshot_decl.location() = location;
    target_block.copy_to_operands(snapshot_decl);
    code_assignt snapshot_copy(symbol_expr(snapshot), symbol_expr(*source));
    snapshot_copy.location() = location;
    target_block.copy_to_operands(snapshot_copy);
  }

  const exprt source_base = python_expr::build_typecast(
    python_expr::build_address_of(symbol_expr(*source)), scalar_ptr);
  const exprt snapshot_base = python_expr::build_typecast(
    python_expr::build_address_of(symbol_expr(snapshot)), scalar_ptr);
  for (const std::string &view_id : view_ids)
  {
    const exprt view_ptr = symbol_expr(*symbol_table_.find_symbol(view_id));
    const exprt offset = python_expr::build_sub(
      python_expr::build_typecast(view_ptr, scalar_ptr),
      source_base,
      signed_size_type());
    code_assignt repoint(
      view_ptr,
      python_expr::build_typecast(
        python_expr::build_add(snapshot_base, offset, scalar_ptr),
        view_ptr.type()));
    repoint.location() = location;
    target_block.copy_to_operands(repoint);

    numpy_pointer_view_info_[view_id].source_id = snapshot.id.as_string();
    numpy_view_copy_sources_.erase(view_id);
  }
}

// `b = a; a = ...`: b keeps the storage a had. The first alias takes a copy
// of it and the other aliases follow that alias; the views of `a` are then
// repointed there too, so all of them keep sharing one storage. The alias
// map is static, so a rebind on one path only is rejected.
std::string python_converter::hand_numpy_storage_to_alias(
  const std::string &rebound_id,
  const locationt &location,
  codet &target_block)
{
  const std::vector<std::string> aliases = numpy_storage_aliases_of(rebound_id);
  const symbolt *source = symbol_table_.find_symbol(rebound_id);
  const symbolt *heir =
    aliases.empty() ? nullptr : symbol_table_.find_symbol(aliases.front());
  if (!source || !heir || heir->get_type() != source->get_type())
    return "";
  if (block_nesting_ != function_body_depth_ + 1 || loop_body_depth_ > 0)
    throw std::runtime_error(
      "TypeError: rebinding a numpy array inside a branch or loop while "
      "another name aliases it is not supported");
  code_assignt copy(symbol_expr(*heir), symbol_expr(*source));
  copy.location() = location;
  target_block.copy_to_operands(copy);
  numpy_array_storage_aliases_.erase(aliases.front());
  for (std::size_t i = 1; i < aliases.size(); ++i)
    numpy_array_storage_aliases_[aliases[i]] = aliases.front();
  return aliases.front();
}

void python_converter::detach_numpy_pointer_views_of(
  const std::string &rebound_id,
  const locationt &location,
  codet &target_block)
{
  const std::string heir =
    hand_numpy_storage_to_alias(rebound_id, location, target_block);
  numpy_nditer(*this).keep_sources(rebound_id, heir);
  std::map<std::string, std::vector<std::string>> views_by_storage;
  for (const std::string &view_id : numpy_views_of(rebound_id))
  {
    const auto info = numpy_pointer_view_info_.find(view_id);
    // Which storage such a view points at is only known at run time, so it
    // cannot be repointed at a snapshot of one of them.
    if (
      info != numpy_pointer_view_info_.end() &&
      !info->second.other_sources.empty())
      throw std::runtime_error(
        "TypeError: rebinding a numpy array while a view bound to different "
        "arrays on different paths is alive is not supported");
    if (
      info != numpy_pointer_view_info_.end() &&
      symbol_table_.find_symbol(view_id))
      views_by_storage[resolve_numpy_array_storage_alias_id(
                         info->second.source_id)]
        .push_back(view_id);
  }
  // A snapshot declared in a nested scope expires with it, and a rebind on
  // one path leaves the views' static metadata wrong on the other.
  if (!views_by_storage.empty() && block_nesting_ != function_body_depth_ + 1)
    throw std::runtime_error(
      "TypeError: rebinding a numpy array inside a branch or loop while "
      "views of it are alive is not supported");
  for (const auto &[storage_id, view_ids] : views_by_storage)
  {
    if (!heir.empty() && storage_id == rebound_id)
      share_numpy_storage_snapshot(
        storage_id, view_ids, location, target_block, heir);
    else if (view_ids.size() == 1)
      detach_numpy_pointer_view(view_ids.front(), location, target_block);
    else
      share_numpy_storage_snapshot(
        storage_id, view_ids, location, target_block);
  }
}
