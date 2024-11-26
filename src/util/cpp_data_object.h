#ifndef ESBMC_CPP_DATA_OBJECT_H
#define ESBMC_CPP_DATA_OBJECT_H

#include "util/std_types.h"
#include "util/namespace.h"
#include <string>
#include <utility>

/**
 * @class cpp_data_object
 * @brief Provides utility functions for handling C++ data objects.
 *
 * A data object in C++ is not an official term, but it is used here to refer to
 * a struct that represents the _non-virtual_ part of a C++ class.
 * Consider the following example:
 *
 * ```
 * class A {
 *  int x;
 * };
 *
 * Then an object for A would contain only a single field,
 * "A@data_object", which is a struct with a single field "x".
 * If there is a class B that inherits from A, then the object
 * for B would contain a single field "B@data_object", which
 * contains a field "A@data_object" nested in it.
 * A class `C` that inherits from `B` and `virtual A` would contain
 * two fields: "B@data_object" and "A@data_object" (for the virtual base).
 * Note: we do some liberal casting between the class "A" and "A@data_object".
 * In a strict C++ sense this is potentially UB, but we are not bound by the
 * C++ standard in our internal representation.
 *
 */
class cpp_data_object
{
public:
  /**
   * @brief Suffix used for data objects.
   *
   * This static inline member holds the suffix string used to identify
   * data objects.
   */
  static inline std::string data_object_suffix = "@data_object";

  /**
   * @brief Retrieves the data object type for a given class name.
   *
   * @param class_name The name of the class for which the data object type is retrieved.
   * @param context The context in which the data object type is defined.
   * @return A reference to the struct type representing the data object type.
   *
   * This static method retrieves the data object type for a specified class
   * name within the given context. It returns a reference to the struct type
   * representing the data object type.
   */
  static struct_typet &
  get_data_object_type(const std::string &class_name, contextt &context);

  /**
   * @brief Retrieves the symbol type for a data object.
   *
   * @param class_name The name of the class for which the data object symbol type is retrieved.
   * @param data_object_symbol_type A reference to the type object where the symbol type will be stored.
   *
   * This static method retrieves the symbol type for a data object corresponding
   * to the specified class name. The symbol type is stored in the provided
   * type object reference.
   */
  static void get_data_object_symbol_type(
    const std::string &class_name,
    typet &data_object_symbol_type);

  /**
   * @brief Retrieves the type that contains the specified data object component.
   * For non-virtual bases, the containing type is the data object type for `type`.
   * For virtual bases, the containing type is `type` itself.
   * E.g. Searching for "B" (non-virtual base) in the type "C" would return {C\@data_object, B\@data_object}
   * while searching for "A" (virtual base) in the type "C" would return {C, A\@data_object}.
   * @param type the struct type to search for the data object.
   * @param data_object_name the name of the data object component to search for without `\@data_object`.
   * @param ns used resolve symbol types.
   * @return a pair containing the type that contains the data object and the data object component.
   */
  static std::pair<const typet &, struct_union_typet::componentt>
  get_data_object(
    const typet &type,
    const dstring &data_object_name,
    const namespacet &ns);
};

#endif //ESBMC_CPP_DATA_OBJECT_H