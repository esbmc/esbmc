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

  // current access specifier
  std::string current_access;
  // current struc or class symbol type
  struct_union_typet *current_class_type;

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
   * Get member initializers for constructor
   */
  bool get_ctor_initializers(code_blockt &body);

  bool get_class_method(
    const clang::FunctionDecl &fd,
    exprt &component,
    code_typet &method_type,
    const symbolt &method_symbol) override;

  bool get_virtual_method(
    const clang::FunctionDecl &fd,
    exprt &component,
    code_typet &method_type,
    const symbolt &method_symbol,
    const std::string &class_symbol_id) override;

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
    struct_union_typet &type) override;

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

  std::string get_access(const clang::AccessSpecifier &specifier);

  void do_virtual_table(const struct_union_typet &type);

  /*
   * Get irept node for virtual pointer initialization
   */
  bool get_vptr_init_irep(exprt &vptr_init, const clang::FunctionDecl &fd);

  /*
   * Get assignment expression to represent vptr initialization
   */
  void get_vptr_init_expr(
    const exprt &vptr_init,
    code_blockt &body,
    const clang::FunctionDecl &fd,
    const code_typet &ftype);

  /*
   * Get access specifier for a method
   */
  void get_current_access(
    const clang::FunctionDecl &target_fd,
    const clang::CXXRecordDecl &cxxrd);

  /*
   * Get parent class symbol for a non-inlined method
   */
  symbolt *get_parent_class_symbol(const clang::FunctionDecl &target_fd);

  /**
   * Typecheck base(s)
   * cxxrd refers to the clang AST of the derived class
   */
  bool get_bases(const clang::RecordDecl &rd, struct_typet &derived_class_type)
    override;

  /**
   * Traverse the class hierarchy and
   * copy components from base(s) to the derived
   */
  void add_base_components(
    const clang::CXXRecordDecl &base_cxxrd,
    const struct_typet &from,
    const clang::AccessSpecifier &access,
    struct_typet &to,
    std::set<irep_idt> &bases,
    std::set<irep_idt> &vbases,
    bool is_virtual);

  /**
   * When typechecking a method in derived class
   * this function populates a set of bases that contain
   * the virtual method.
   */
    void get_method_virtual_bases(
        std::set<irep_idt> &virtual_bases,
      const std::string &class_symbol_id,
      const std::string &virtual_name);
};

#endif /* CLANG_C_FRONTEND_CLANG_C_CONVERT_H_ */
