#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_CONVERT_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_CONVERT_H_

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include <clang-c-frontend/clang_c_convert.h>

class clang_cpp_convertert : public clang_c_convertert
{
public:
  clang_cpp_convertert(
    contextt &_context,
    std::vector<std::unique_ptr<clang::ASTUnit>> &_ASTs,
    irep_idt _mode);
  virtual ~clang_cpp_convertert() = default;

protected:
  // this_map contains key-value pairs in the form of <method address, <identifier, type>>
  typedef std::unordered_map<std::size_t, std::pair<std::string, typet>>
    this_mapt;
  this_mapt this_map;

  bool get_decl(const clang::Decl &decl, exprt &new_expr) override;

  /**
   *  Get reference to declared variabales or functions, e.g:
   *   - getting the callee for CXXConstructExpr
   *   - getting the object for DeclRefExpr
   */
  bool get_decl_ref(const clang::Decl &decl, exprt &new_expr);

  bool get_type(const clang::QualType &type, typet &new_type) override;

  bool get_type(const clang::Type &the_type, typet &new_type) override;

  bool get_function(const clang::FunctionDecl &fd, exprt &new_expr) override;

  /**
   *  Get reference for constructor callsite
   */
  bool get_constructor_call(
    const clang::CXXConstructExpr &constructor_call,
    exprt &new_expr);

  bool get_function_body(
    const clang::FunctionDecl &fd,
    exprt &new_expr,
    const code_typet &ftype) override;

  /**
   *  Get function params for C++
   *  contains parsing routines specific to C++ class member functions
   */
  bool get_function_params(
    const clang::FunctionDecl &fd,
    code_typet::argumentst &params) override;

  /**
   *  Add implicit `this' when parsing C++ class member functions, e.g:
   *  class t1 { int i; t1(){i = 1} };
   *  The implicit `this' within the constructor is represented by symbol:
   *   - t1() symbol.type:   void (t1*)
   *   - t1() symbol.value:  { t1->i = 1; }
   */
  bool get_function_this_pointer_param(
    const clang::CXXMethodDecl &fd,
    code_typet::argumentst &params);

  bool get_struct_union_class(const clang::RecordDecl &rd) override;

  bool get_var(const clang::VarDecl &vd, exprt &new_expr) override;

  bool get_struct_union_class_fields(
    const clang::RecordDecl &rd,
    struct_union_typet &type) override;

  bool get_struct_union_class_methods(
    const clang::RecordDecl &rd,
    struct_typet &type) override;

  template <typename TemplateDecl>
  bool get_template_decl(
    const TemplateDecl *D,
    bool DumpExplicitInst,
    exprt &new_expr);

  template <typename SpecializationDecl>
  bool get_template_decl_specialization(
    const SpecializationDecl *D,
    bool DumpExplicitInst,
    bool DumpRefOnly,
    exprt &new_expr);

  bool get_expr(const clang::Stmt &stmt, exprt &new_expr) override;

  void
  build_member_from_component(const clang::FunctionDecl &fd, exprt &component);

  /*
   * Add access in class symbol type's component:
   *  0: component:
   *    * access: public
   */
  bool annotate_class_field_access(
    const clang::FieldDecl *field,
    struct_typet::componentt &comp);

  /*
   * Get access from any clang Decl (field, method .etc)
   */
  bool get_access_from_decl(const clang::Decl *decl, std::string &access);

  /*
   * Add additional annotations for class/struct/union methods:
   * 0: component
   *    * type: code
   *      * #member_name: tag-MyClasss
   *      * return_type: constructor
   *    * from_base: 1
   *    * access: public
   *    * name: MyMethodName
   *    * #location:
   * Arguments:
   *  cxxmd: clang AST node representing the method we are dealing with
   *  new_expr: the `component` in class/struct/union symbol type
   */
  bool
  annotate_cpp_methods(const clang::CXXMethodDecl *cxxmdd, exprt &new_expr);

  /*
   * When getting a function call to ctor, we might call the base ctor from a derived class ctor
   * Need to wrap derived class `this` in a typecast expr and convert to the base `this`, e.g.:
   *    Base( (Base*) this)
   *
   * Arguments:
   *  callee_decl: base ctor symbol
   *  call: function call statement to the base ctor
   *  initializer: this is an intermediate data structure containing the information of derived `this`
   */
  void gen_typecast_base_ctor_call(
    const exprt &callee_decl,
    side_effect_expr_function_callt &call,
    exprt &initializer);

  /*
   * This is an ancillary function deciding whether we need
   * need to build an new_object when dealing with constructor_call
   */
  bool need_new_object(const clang::Stmt *parentStmt);

  /*
   * Functions for virtual tables and virtual pointers
   *  TODO: add link to wiki page
   */
  std::string vtable_type_prefix = "virtual_table::";
  /*
   * traverse methods to:
   *  1. convert virtual methods and add them to class' type
   *  2. If there is no existing vtable type symbol,
   *    add virtual table type symbol in the context
   *    and the virtual pointer.
   *    Then add a new entry in the vtable.
   *  3. If vtable type symbol exists, do not add it or the vptr,
   *    just add a new entry in the existing vtable.
   */
  bool get_struct_class_virtual_methods(
    const clang::CXXRecordDecl *cxxrd,
    struct_typet &type);
  /*
   * additional annotations for virtual or overriding methods
   */
  bool annotate_virtual_overriding_methods(
    const clang::CXXMethodDecl *md,
    struct_typet::componentt &comp);
  /*
   * Check the existence of virtual table type symbol.
   * If it exists, return its pointer. Otherwise, return nullptr.
   */
  symbolt *check_vtable_type_symbol_existence(struct_typet &type);
  /*
   * Add virtual table(vtable) type symbol.
   * This is added as a type symbol in the symbol table.
   *
   * This is done the first time when we encounter a virtual method in a class
   */
  symbolt *add_vtable_type_symbol(
    const clang::CXXMethodDecl *md,
    const struct_typet::componentt &comp,
    struct_typet &type);
  /*
   * Add virtual pointer(vptr).
   * Vptr is NOT a symbol but rather simply added as a component to the class' type.
   *
   * This is done the first time we encounter a virtual method in a class
   */
  void add_vptr(const clang::CXXMethodDecl *md, struct_typet &type);
  /*
   * Add an entry to the virtual table type
   *
   * This is done when NOT the first time we encounter a virtual method in a class
   * in which case we just want to add a new entry to the virtual table type
   */
  void add_vtable_type_entry(
    struct_typet &type,
    struct_typet::componentt &comp,
    symbolt *vtable_type_symbol);
};

#endif /* CLANG_C_FRONTEND_CLANG_C_CONVERT_H_ */
