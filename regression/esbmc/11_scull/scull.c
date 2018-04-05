#include "scull.h"
#include <pthread.h> 
/* =====================================================
   User program calling functions from the device driver
   ===================================================== */
inode i;
int lock = FILE_WITH_LOCK_UNLOCKED;
int NONDET;

void loader() {
  scull_init_module();
  scull_cleanup_module();
}

void thread1() {
  file filp;
  char buf;
  size_t count = 10;
  loff_t off = 0;
  scull_open(tid1, i, filp);
  scull_read(tid1, filp, buf, count, off);
  scull_release(i, filp);
}

void thread2() {
  file filp;
  char buf;
  size_t count = 10;
  loff_t off = 0;
  scull_open(tid2, i, filp);
  scull_write(tid2, filp, buf, count, off);
  scull_release(i, filp);
}

/* =====================================================
   Model for the Linux kernel API
   ===================================================== */
#define acquire_thread_id(tid, l) \
  { __blockattribute__((atomic)) \
    assume(l==0); \
    l = tid; \
  } \
 
inline int down_interruptible(int tid) {
  acquire_thread_id(tid, lock);
  return 0; // lock is held
}

#define up() release(lock)

#define container_of(dev) dev

inline unsigned_long copy_to_user(char to, char from, unsigned_long n) {
  to = from;
  return NONDET;
}

inline unsigned_long copy_from_user(char to, char from, unsigned_long n) {
  to = from;
  return NONDET;
}

inline int __get_user(int size, void_ptr ptr)
{
  return NONDET;
}

inline int __put_user(int size, void_ptr ptr)
{
    return NONDET;
} 


/* =====================================================
   A model for the device-driver functions
   ===================================================== */
/*
 * scull.h -- definitions for the char module
 *
 * Copyright (C) 2001 Alessandro Rubini and Jonathan Corbet
 * Copyright (C) 2001 O'Reilly & Associates
 *
 * The source code in this file can be freely used, adapted,
 * and redistributed in source or binary form, so long as an
 * acknowledgment appears in derived source files.  The citation
 * should list that the code comes from the book "Linux Device
 * Drivers" by Alessandro Rubini and Jonathan Corbet, published
 * by O'Reilly & Associates.   No warranty is attached;
 * we cannot take responsibility for errors or fitness for use.
 *
 * $Id: scull.h,v 1.15 2004/11/04 17:51:18 rubini Exp $
 */

int scull_quantum = SCULL_QUANTUM;
int scull_qset = SCULL_QSET;
int dev_data;
int dev_quantum;
int dev_qset;
unsigned_long dev_size; 
int __X__; //variable to test mutual exclusion 

/*
 * Empty out the scull device; must be called with the device
 * semaphore held.
 */
int scull_trim(scull_dev dev)
{
  int qset = dev_qset;

  dev_size = 0;
  dev_quantum = scull_quantum;
  dev_qset = scull_qset;
  dev_data = NULL;
  return 0;
}


/*
 * Open and close
 */

inline int scull_open(int tid, inode i, file filp) 
{
  scull_dev dev;

  dev = container_of(i);
  filp = dev; // filp->private_data = dev;

  if (down_interruptible(tid))
    return -ERESTARTSYS;

  __X__ = 2;          /* check mutual exclusion */
  scull_trim(dev); /* ignore errors */
  assert(__X__ >= 2); /* check mutual exclusion */
  up();
  return 0;          /* success */
}

#define scull_release(i, filp) 0

/*
 * Follow the list
 */
inline scull_qset_type scull_follow(scull_dev dev, int n) {
  return NONDET;
}

/*
 * Data management: read and write
 */

inline ssize_t scull_read(int tid, file filp, char buf, size_t count, 
			  loff_t f_pos) 
{
  scull_dev dev = filp; //struct scull_dev *dev = filp->private_data
  scull_qset_type dptr; /* the first listitem */
  int quantum = dev_quantum, qset = dev_qset;
  int itemsize = quantum * qset; /* how many bytes in the listitem */
  int item, s_pos, q_pos, rest;
  ssize_t retval = 0;

  if (down_interruptible(tid))
    return -ERESTARTSYS;

  __X__ = 0;          /* check mutual exclusion */

  if (f_pos >= dev_size) 
    goto out;
  if (f_pos+count >= dev_size)
    count = dev_size - f_pos;

  /* find listitem, qset index, and offset in the quantum */
  item = f_pos / itemsize; 
  rest = f_pos; 
   s_pos = rest / quantum; q_pos = rest;

   /* follow the list up to the right position (defined elsewhere) */
   dptr = scull_follow(dev, item);

   /* read only up to the end of this quantum */ 
   if (count > quantum - q_pos) 
     count = quantum - q_pos; 
  
  if (copy_to_user(buf, dev_data + s_pos + q_pos, count)) {
    retval = -EFAULT;
    goto out;
  }
  f_pos += count;
  retval = count;

  assert(__X__ <= 0); /* check mutual exclusion */

 out:
  up();
  return retval;
}

inline ssize_t scull_write(int tid, file filp, char buf, size_t count, 
			   loff_t f_pos) 
{
  scull_dev dev = filp; //struct scull_dev *dev = filp->private_data
  scull_qset_type dptr;
  int quantum = dev_quantum, qset = dev_qset;
  int itemsize = quantum * qset;
  int item, s_pos, q_pos, rest;
  ssize_t retval = -ENOMEM; /* value used in "goto out" statements */

  if (down_interruptible(tid))
    return -ERESTARTSYS;
  
  /* find listitem, qset index and offset in the quantum */
  item = f_pos / itemsize;
  rest = f_pos;
  s_pos = rest / quantum; q_pos = rest;

  /* follow the list up to the right position */
  dptr = scull_follow(dev, item);
  if (dptr == NULL)
    goto out;

  /* write only up to the end of this quantum */
  if (count > quantum - q_pos)
    count = quantum - q_pos;

  __X__ = 1;          /* check mutual exclusion */

  if (copy_from_user(dev_data+s_pos+q_pos, buf, count)) {
    retval = -EFAULT;
    goto out;
  }
  f_pos += count;
  retval = count;

  /* update the size */
  if (dev_size < f_pos)
    dev_size = f_pos;

  assert(__X__ == 1); /* check mutual exclusion */

 out:
  up();
  return retval;
}

/*
 * The ioctl() implementation
 */

inline int scull_ioctl(inode i, file filp,
                 unsigned_int cmd, unsigned_long arg)
{

	int err = 0, tmp;
	int retval = 0;
    
	switch(cmd) {

	  case SCULL_IOCRESET:
		scull_quantum = SCULL_QUANTUM;
		scull_qset = SCULL_QSET;
		break;
        
	  case SCULL_IOCSQUANTUM: /* Set: arg points to the value */
		retval = __get_user(scull_quantum, arg);
		break;

	  case SCULL_IOCTQUANTUM: /* Tell: arg is the value */
		scull_quantum = arg;
		break;

	  case SCULL_IOCGQUANTUM: /* Get: arg is pointer to result */
		retval = __put_user(scull_quantum, arg);
		break;

	  case SCULL_IOCQQUANTUM: /* Query: return it (it's positive) */
		return scull_quantum;

	  case SCULL_IOCXQUANTUM: /* eXchange: use arg as pointer */
		tmp = scull_quantum;
		retval = __get_user(scull_quantum, arg);
		if (retval == 0)
			retval = __put_user(tmp, arg);
		break;

	  case SCULL_IOCHQUANTUM: /* sHift: like Tell + Query */
		tmp = scull_quantum;
		scull_quantum = arg;
		return tmp;
        
	  case SCULL_IOCSQSET:
		retval = __get_user(scull_qset, arg);
		break;

	  case SCULL_IOCTQSET:
		scull_qset = arg;
		break;

	  case SCULL_IOCGQSET:
		retval = __put_user(scull_qset, arg);
		break;

	  case SCULL_IOCQQSET:
		return scull_qset;

	  case SCULL_IOCXQSET:
		tmp = scull_qset;
		retval = __get_user(scull_qset, arg);
		if (retval == 0)
			retval = put_user(tmp, arg);
		break;

	  case SCULL_IOCHQSET:
		tmp = scull_qset;
		scull_qset = arg;
		return tmp;


	  default:  /* redundant, as cmd was checked against MAXNR */
		return -ENOTTY;
	}
	return retval;

}


/*
 * The "extended" operations -- only seek
 */

inline loff_t scull_llseek(file filp, loff_t off, int whence)
{
  scull_dev dev = filp; // dev->private_data;
  loff_t newpos;

  switch(whence) {
  case 0: /* SEEK_SET */
    newpos = off;
    break;

  case 1: /* SEEK_CUR */
    newpos = filp + f_pos + off;
    break;

  case 2: /* SEEK_END */
    newpos = dev_size + off;
    break;

  default: /* can't happen */
    return -EINVAL;
  }
  if (newpos < 0) return -EINVAL;
  filp = newpos;
  return newpos;
}

/*
 * Finally, the module stuff
 */

/*
 * The cleanup function is used to handle initialization failures as well.
 * Thefore, it must be careful to work correctly even if some of the items
 * have not been initialized
 */
inline void scull_cleanup_module(void) 
{
  scull_dev dev;
  scull_trim(dev);

}

inline int scull_init_module() 
{
  int result = 0;
  return 0;

 fail:
  scull_cleanup_module();
  return result;
}

int main() {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, thread1, NULL);
  pthread_create(&t2, NULL, thread12, NULL);
  return 0;
}
