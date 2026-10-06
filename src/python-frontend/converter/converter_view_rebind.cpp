#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>

#include <util/lang/c_types.h>

#include <map>
#include <stdexcept>

// Rebinding a NumPy array name leaves the old array, and every view of it,
// untouched. Views of one storage are repointed together into one snapshot
// of that storage, so they keep aliasing each other.

void python_converter::share_numpy_storage_snapshot(
  const std::string &storage_id,
  const std::vector<std::string> &view_ids,
  const locationt &location,
  codet &target_block)
{
  const namespacet ns(symbol_table_);
  const symbolt *source = symbol_table_.find_symbol(storage_id);
  const typet source_type = source ? ns.follow(source->get_type()) : typet();
  if (
    !source_type.is_array() || !to_array_type(source_type).size().is_constant())
    throw std::runtime_error(
      "TypeError: sibling numpy views of a rebound source need a fixed-size "
      "array to share a snapshot");

  typet scalar_type = source_type;
  while (scalar_type.is_array())
    scalar_type = ns.follow(to_array_type(scalar_type).subtype());
  const pointer_typet scalar_ptr(scalar_type);

  symbolt &snapshot =
    create_tmp_symbol(location, "$rebind_snapshot$", source_type, exprt());
  code_declt snapshot_decl(symbol_expr(snapshot));
  snapshot_decl.location() = location;
  target_block.copy_to_operands(snapshot_decl);
  code_assignt snapshot_copy(symbol_expr(snapshot), symbol_expr(*source));
  snapshot_copy.location() = location;
  target_block.copy_to_operands(snapshot_copy);

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

void python_converter::detach_numpy_pointer_views_of(
  const std::string &rebound_id,
  const locationt &location,
  codet &target_block)
{
  std::map<std::string, std::vector<std::string>> views_by_storage;
  for (const std::string &view_id : numpy_views_of(rebound_id))
  {
    const auto info = numpy_pointer_view_info_.find(view_id);
    if (
      info != numpy_pointer_view_info_.end() &&
      symbol_table_.find_symbol(view_id))
      views_by_storage[resolve_numpy_array_storage_alias_id(
                         info->second.source_id)]
        .push_back(view_id);
  }
  for (const auto &[storage_id, view_ids] : views_by_storage)
  {
    if (view_ids.size() == 1)
      detach_numpy_pointer_view(view_ids.front(), location, target_block);
    else
      share_numpy_storage_snapshot(
        storage_id, view_ids, location, target_block);
  }
}
