typedef long int ptrdiff_t;
typedef struct UT_hash_bucket
{
  struct UT_hash_handle *hh_head;
  unsigned count;
} UT_hash_bucket;
typedef struct UT_hash_table
{
  UT_hash_bucket *buckets;
  unsigned num_buckets, log2_num_buckets;
  unsigned num_items;
  struct UT_hash_handle *tail;
  ptrdiff_t hho;
} UT_hash_table;
typedef struct UT_hash_handle
{
  struct UT_hash_table *tbl;
  void *prev;
  void *next;
  struct UT_hash_handle *hh_prev;
  struct UT_hash_handle *hh_next;
  unsigned hashv;
} UT_hash_handle;
typedef struct example_user_t
{
  UT_hash_handle hh;
} example_user_t;
int main()
{
  int i;
  example_user_t *user, *tmp, *users = ((void *)0);
  for (i = 0; i < 10; i += 2)
  {
    if (tmp != ((void *)0))
    {
      do
      {
        struct UT_hash_handle *_hd_hh_del = (&(tmp)->hh);
        if (
          (_hd_hh_del->prev == ((void *)0)) &&
          (_hd_hh_del->next == ((void *)0)))
        {
          ;
          free((users)->hh.tbl->buckets);
          free((users)->hh.tbl);
          (users) = ((void *)0);
        }
        else
        {
          unsigned _hd_bkt;
          if (_hd_hh_del == (users)->hh.tbl->tail)
          {
            (users)->hh.tbl->tail =
              ((UT_hash_handle
                  *)(((char *)(_hd_hh_del->prev)) + (((users)->hh.tbl)->hho)));
          }
          if (_hd_hh_del->prev != ((void *)0))
          {
            ((UT_hash_handle
                *)(((char *)(_hd_hh_del->prev)) + (((users)->hh.tbl)->hho)))
              ->next = _hd_hh_del->next;
          }
          else
          {
            do
            {
              (users) = (__typeof(users))(_hd_hh_del->next);
            } while (0);
          }
          if (_hd_hh_del->next != ((void *)0))
          {
            ((UT_hash_handle
                *)(((char *)(_hd_hh_del->next)) + (((users)->hh.tbl)->hho)))
              ->prev = _hd_hh_del->prev;
          }
          do
          {
            _hd_bkt =
              ((_hd_hh_del->hashv) & (((users)->hh.tbl->num_buckets) - 1U));
          } while (0);
          do
          {
            UT_hash_bucket *_hd_head = &((users)->hh.tbl->buckets[_hd_bkt]);
            _hd_head->count--;
            if (_hd_head->hh_head == (_hd_hh_del))
            {
              _hd_head->hh_head = (_hd_hh_del)->hh_next;
            }
            if ((_hd_hh_del)->hh_prev)
            {
              (_hd_hh_del)->hh_prev->hh_next = (_hd_hh_del)->hh_next;
            }
            if ((_hd_hh_del)->hh_next)
            {
              (_hd_hh_del)->hh_next->hh_prev = (_hd_hh_del)->hh_prev;
            }
          } while (0);
          (users)->hh.tbl->num_items--;
        }
      } while (0);
    }
  }
  for (user = users; user != ((void *)0);
       user = (example_user_t *)(user->hh.next))
  {
  }
}
