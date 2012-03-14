
/** Base class for all types */
class type2t
{
  enum type_ids {
  };
  type_ids type_id;
};

/** Base class for all expressions */
class expr2t
{
  enum expr_ids {
  };
  expr_ids expr_id;

  type2t type;
};
