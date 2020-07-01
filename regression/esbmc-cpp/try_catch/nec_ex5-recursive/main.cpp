#include <cstdio>

class BlockParseException
{
public:
  BlockParseException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class VarDeclParseException
{
public:
  VarDeclParseException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class SingleStmtParseException
{
public:
  SingleStmtParseException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class CompoundStmtParseException
{
public:
  CompoundStmtParseException ()
  {
    goto ERROR;
    ERROR:
    ;
  }
};

class SingleStmt;
class CompoundStmt;
class VarDecl;

class CompoundStmt {
public:
  SingleStmt** stmts;
  int numStmts;

  CompoundStmt(int ns)
  {
    numStmts = ns;
    stmts = new SingleStmt * [numStmts];
  }

  ~CompoundStmt()
  {
    delete[] stmts;
  }
};

class Block {
public:
  int numVarDecls;
  int numCompStmts;
  VarDecl** varDecls;
  CompoundStmt** compoundStmts;

  Block(int nV, int nC)
  {
    numVarDecls = nV;
    numCompStmts = nC;
    varDecls = new VarDecl * [nV];
    compoundStmts = new CompoundStmt * [nC];
  }

  ~Block()
  {
    delete[] varDecls;
    delete[] compoundStmts;
  }
};

class VarDecl {
public:
  int id;
  VarDecl(int i)
  {
    id = i;
  }
  ~VarDecl()
  {
  }
};


class SingleStmt {
public:
  bool isBlock;
  Block* b;
  SingleStmt()
  {
    isBlock = false;
  }
  ~SingleStmt() {
  }
};

void parseVarDecl(VarDecl* v);
void parseCompoundStmt(CompoundStmt* cStmt);
void parseSingleStmt(SingleStmt* sStmt);

void parseBlock(Block* b)
{
  try {
    if (b == NULL)
      throw BlockParseException();

    for (int i=0; i < b->numVarDecls; ++i)
      parseVarDecl(b->varDecls[i]);

    for (int j=0; j < b->numCompStmts; ++j)
      parseCompoundStmt(b->compoundStmts[j]);
  }
  catch (VarDeclParseException& v)
  {
    printf("Caught VariableDeclException\n");
  }
  catch (SingleStmtParseException& s)
  {
    printf("Caught SingleStmtParseException\n");
  }
}

void parseVarDecl(VarDecl* v)
{
  if (v == NULL)
    throw VarDeclParseException();
  // Parse code
}

void parseCompoundStmt(CompoundStmt* cStmt)
{
  try {
    if (cStmt == NULL)
      throw CompoundStmtParseException();

    for (int i=0; i < cStmt->numStmts; ++i)
    {
      parseSingleStmt(cStmt->stmts[i]);
    }
  }
  catch (BlockParseException& b)
  {
    printf("Caught Block Parse Exception\n");
  }
}

void parseSingleStmt(SingleStmt* sStmt)
{
  try {
    if (sStmt == NULL)
      throw SingleStmtParseException();

    if (sStmt->isBlock)
      parseBlock(sStmt->b);
  }
  catch (CompoundStmtParseException& e)
  {
    printf("Caught CompoundStmtParseException\n");
  }
}

int main()
{
  VarDecl* vd = NULL; //new VarDecl(0);
  Block* b1 = new Block(1,0);
  b1->varDecls[0] = vd;

  SingleStmt* ss = new SingleStmt();
  ss->isBlock = true;
  ss->b = b1;

  CompoundStmt* cStmt = new CompoundStmt(1);
  cStmt->stmts[0] = ss;

  Block* b = new Block(0, 1);
  b->compoundStmts[0] = cStmt;

  parseBlock(b);
  printf("Created all stmts\n");
  delete vd;
  delete b1;
  delete ss;
  delete cStmt;
  delete b;

  return 0;
}
