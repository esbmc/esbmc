#ifndef _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_
#define _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_

class crypto_hash_private;

#include <string>
#include <memory>

class crypto_hash_private;

#define CRYPTO_HASH_SIZE  32

class crypto_hash {
public:
  uint8_t hash[32];
  std::shared_ptr<crypto_hash_private> p_crypto;

  bool operator<(const crypto_hash h2) const;

  std::string to_string() const;

  crypto_hash();
  void ingest(void const *data, unsigned int size);
  void fin();
};

#endif /* _CPROVER_SRC_GOTO_SYMEX_CRYPTO_HASH_H_ */
