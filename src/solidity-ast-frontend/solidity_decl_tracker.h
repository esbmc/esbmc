#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_

#include <nlohmann/json.hpp>
#include <solidity-ast-frontend/solidity_type.h>
#include <iostream>
#include <iomanip>
#include <vector>

class decl_function_tracker
{
public:
  // invalid values to be more defensive
  static constexpr unsigned plocLineInvalid = std::numeric_limits<unsigned>::max();
  static constexpr size_t numParamInvalid = std::numeric_limits<size_t>::max();
  static constexpr size_t paramIndexInvalid = std::numeric_limits<size_t>::max();

  decl_function_tracker(nlohmann::json& _decl_func) :
    decl_func(_decl_func)
  {
      clear_all();
  }

  decl_function_tracker(const decl_function_tracker &rhs) = default; // default copy constructor
  // move constructor
  decl_function_tracker(decl_function_tracker &&rhs) = default;
  // copy assignment operator - TODO: Since we have a reference member, it can be tricky in this case.
  //    Let's not do it for the time being. Leave it for future improvement
  decl_function_tracker& operator=(const decl_function_tracker &rhs) { assert(!"copy assignment is not allowed at the moment"); }
  // move assignment operator - TODO: Since we have a reference member, it can be tricky in this case.
  //    Let's not do it for the time being. Leave it for future improvement
  decl_function_tracker& operator=(decl_function_tracker &&rhs) { assert(!"move assignment is not allowed at the moment"); }

  void clear_all() // reset to default values
  {
    isImplicit = false;
    isDefined = false;
    isThisDeclarationADefinition = false;
    type_class = SolidityTypes::TypeError;
    decl_class = SolidityTypes::DeclError;
    builtin_type = SolidityTypes::BuiltInError;
    storage_class = SolidityTypes::SCError;
    isConstQualified = false;
    isVolatileQualified = false;
    isRestrictQualified = false;
    isVariadic = false;
    isInlined = false;
    isFunctionOrMethod = false;
    PLoc_Line = plocLineInvalid;
    declName = "";
    id = "";
    moduleName = "";
    num_param = numParamInvalid;
  }

  // for debug print
  void print_decl_func_json();

  // config this tracker based on json values
  void config();

  // getters
  bool get_isImplicit() { return isImplicit; }
  bool get_isDefined() { return isDefined; }
  bool get_isThisDeclarationADefinition() { return isThisDeclarationADefinition; }
  bool get_isConstQualified() { return isConstQualified; }
  bool get_isVolatileQualified() { return isVolatileQualified; }
  bool get_isRestrictQualified() { return isRestrictQualified; }
  bool get_isVariadic() { return isVariadic; }
  bool get_isInlined() { return isInlined; }
  bool get_isFunctionOrMethod() { return isFunctionOrMethod; }
  unsigned get_ploc_line() { return PLoc_Line; };
  size_t get_num_param() { return num_param; }
  SolidityTypes::typeClass getTypeClass() { return type_class; }
  SolidityTypes::declClass getDeclClass() { return decl_class; }
  SolidityTypes::builInTypes getBuiltInType() { return builtin_type; }
  SolidityTypes::storageClass get_storage_class() { return storage_class; }
  std::string get_declName() { return declName; }
  std::string get_id() { return id; };
  std::string get_moduleName() { return moduleName; }

  // struct to hold parameters
  typedef struct funcParam
  {
    // members to flatten the clang::ParmVarDecl
    SolidityTypes::typeClass type_class;
    SolidityTypes::declClass decl_class;
    SolidityTypes::builInTypes builtin_type;
    bool isConstQualified;
    bool isVolatileQualified;
    bool isRestrictQualified;
    bool isArray;
    bool nameEmpty;
    size_t paramIndex;

    funcParam() { clear(); }
    void print_func_param();

    // explicit copy constructor for push_back
    funcParam(const funcParam &rhs)
    {
      //printf("\t @@ funcParam copy constructor is getting called ... \n");
      type_class = rhs.type_class;
      decl_class = rhs.decl_class;
      builtin_type = rhs.builtin_type;
      isConstQualified = rhs.isConstQualified;
      isVolatileQualified = rhs.isVolatileQualified;
      isRestrictQualified = rhs.isRestrictQualified;
      isArray = rhs.isArray;
      nameEmpty = rhs.nameEmpty;
      paramIndex = rhs.paramIndex;
    }

    void clear()
    {
      type_class = SolidityTypes::TypeError;
      decl_class = SolidityTypes::DeclError;
      builtin_type = SolidityTypes::BuiltInError;
      isConstQualified = false;
      isVolatileQualified = false;
      isRestrictQualified = false;
      isArray = false;
      nameEmpty = false;
      paramIndex = paramIndexInvalid;
    }
  } funcParam;
  std::vector<funcParam> params;
  funcParam* get_function_param(unsigned index);

private:
  // TODO: nlohmann_json has explicit copy constructor defined. probably better to just make a copy instead of using a ref member.
  //       makes it easier to set copy assignment operator and move assignment operator for the future.
  nlohmann::json& decl_func;
  bool isImplicit;
  bool isDefined;
  bool isThisDeclarationADefinition;
  bool isConstQualified;
  bool isVolatileQualified;
  bool isRestrictQualified;
  bool isVariadic;
  bool isInlined;
  bool isFunctionOrMethod;
  unsigned PLoc_Line;
  size_t num_param;
  SolidityTypes::typeClass type_class;
  SolidityTypes::declClass decl_class;
  SolidityTypes::builInTypes builtin_type;
  SolidityTypes::storageClass storage_class;
  std::string declName;
  std::string id;
  std::string moduleName;

  // private setters : set the member values based on the corresponding json value. Used by config() only.
  // Setting them outside this class is NOT allowed.
  void set_isImplicit();
  void set_isDefined();
  void set_isThisDeclarationADefinition();
  void set_type_class();
  void set_decl_class();
  void set_builtin_type();
  void set_isConstQualified();
  void set_isVolatileQualified();
  void set_isRestrictQualified();
  void set_isVariadic();
  void set_isInlined();
  void set_isFunctionOrMethod();
  void set_ploc_line();
  void set_declName();
  void set_id();
  void set_moduleName();
  void set_storageClass();
  void set_num_param();

  // populate params if num_param is non-zero
  void populate_params();
};

#endif // END of SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
