#ifndef _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_
#define _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_

#include <memory>
#include <string>

class crypto_hash_private;

#define CRYPTO_HASH_SIZE 32

class crypto_hash
{
public:
  std::shared_ptr<crypto_hash_private> p_crypto;
  unsigned int hash[5];

  bool operator<(const crypto_hash h2) const;

  std::string to_string() const;

  crypto_hash();
  void ingest(void const *data, unsigned int size);
  void fin();
};

#endif /* _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_ */
