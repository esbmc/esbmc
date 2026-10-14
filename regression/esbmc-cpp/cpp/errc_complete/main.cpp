// Every std::errc enumerator must equal its <errno.h> macro; the two are
// generated from one table and must not drift.
#include <system_error>
#include <cerrno>
#include <cassert>

static_assert(static_cast<int>(std::errc::address_family_not_supported) == EAFNOSUPPORT, "address_family_not_supported");
static_assert(static_cast<int>(std::errc::address_in_use) == EADDRINUSE, "address_in_use");
static_assert(static_cast<int>(std::errc::address_not_available) == EADDRNOTAVAIL, "address_not_available");
static_assert(static_cast<int>(std::errc::already_connected) == EISCONN, "already_connected");
static_assert(static_cast<int>(std::errc::argument_list_too_long) == E2BIG, "argument_list_too_long");
static_assert(static_cast<int>(std::errc::argument_out_of_domain) == EDOM, "argument_out_of_domain");
static_assert(static_cast<int>(std::errc::bad_address) == EFAULT, "bad_address");
static_assert(static_cast<int>(std::errc::bad_file_descriptor) == EBADF, "bad_file_descriptor");
static_assert(static_cast<int>(std::errc::bad_message) == EBADMSG, "bad_message");
static_assert(static_cast<int>(std::errc::broken_pipe) == EPIPE, "broken_pipe");
static_assert(static_cast<int>(std::errc::connection_aborted) == ECONNABORTED, "connection_aborted");
static_assert(static_cast<int>(std::errc::connection_already_in_progress) == EALREADY, "connection_already_in_progress");
static_assert(static_cast<int>(std::errc::connection_refused) == ECONNREFUSED, "connection_refused");
static_assert(static_cast<int>(std::errc::connection_reset) == ECONNRESET, "connection_reset");
static_assert(static_cast<int>(std::errc::cross_device_link) == EXDEV, "cross_device_link");
static_assert(static_cast<int>(std::errc::destination_address_required) == EDESTADDRREQ, "destination_address_required");
static_assert(static_cast<int>(std::errc::device_or_resource_busy) == EBUSY, "device_or_resource_busy");
static_assert(static_cast<int>(std::errc::directory_not_empty) == ENOTEMPTY, "directory_not_empty");
static_assert(static_cast<int>(std::errc::executable_format_error) == ENOEXEC, "executable_format_error");
static_assert(static_cast<int>(std::errc::file_exists) == EEXIST, "file_exists");
static_assert(static_cast<int>(std::errc::file_too_large) == EFBIG, "file_too_large");
static_assert(static_cast<int>(std::errc::filename_too_long) == ENAMETOOLONG, "filename_too_long");
static_assert(static_cast<int>(std::errc::function_not_supported) == ENOSYS, "function_not_supported");
static_assert(static_cast<int>(std::errc::host_unreachable) == EHOSTUNREACH, "host_unreachable");
static_assert(static_cast<int>(std::errc::identifier_removed) == EIDRM, "identifier_removed");
static_assert(static_cast<int>(std::errc::illegal_byte_sequence) == EILSEQ, "illegal_byte_sequence");
static_assert(static_cast<int>(std::errc::inappropriate_io_control_operation) == ENOTTY, "inappropriate_io_control_operation");
static_assert(static_cast<int>(std::errc::interrupted) == EINTR, "interrupted");
static_assert(static_cast<int>(std::errc::invalid_argument) == EINVAL, "invalid_argument");
static_assert(static_cast<int>(std::errc::invalid_seek) == ESPIPE, "invalid_seek");
static_assert(static_cast<int>(std::errc::io_error) == EIO, "io_error");
static_assert(static_cast<int>(std::errc::is_a_directory) == EISDIR, "is_a_directory");
static_assert(static_cast<int>(std::errc::message_size) == EMSGSIZE, "message_size");
static_assert(static_cast<int>(std::errc::network_down) == ENETDOWN, "network_down");
static_assert(static_cast<int>(std::errc::network_reset) == ENETRESET, "network_reset");
static_assert(static_cast<int>(std::errc::network_unreachable) == ENETUNREACH, "network_unreachable");
static_assert(static_cast<int>(std::errc::no_buffer_space) == ENOBUFS, "no_buffer_space");
static_assert(static_cast<int>(std::errc::no_child_process) == ECHILD, "no_child_process");
static_assert(static_cast<int>(std::errc::no_link) == ENOLINK, "no_link");
static_assert(static_cast<int>(std::errc::no_lock_available) == ENOLCK, "no_lock_available");
static_assert(static_cast<int>(std::errc::no_message) == ENOMSG, "no_message");
static_assert(static_cast<int>(std::errc::no_protocol_option) == ENOPROTOOPT, "no_protocol_option");
static_assert(static_cast<int>(std::errc::no_space_on_device) == ENOSPC, "no_space_on_device");
static_assert(static_cast<int>(std::errc::no_such_device) == ENODEV, "no_such_device");
static_assert(static_cast<int>(std::errc::no_such_device_or_address) == ENXIO, "no_such_device_or_address");
static_assert(static_cast<int>(std::errc::no_such_file_or_directory) == ENOENT, "no_such_file_or_directory");
static_assert(static_cast<int>(std::errc::no_such_process) == ESRCH, "no_such_process");
static_assert(static_cast<int>(std::errc::not_a_directory) == ENOTDIR, "not_a_directory");
static_assert(static_cast<int>(std::errc::not_a_socket) == ENOTSOCK, "not_a_socket");
static_assert(static_cast<int>(std::errc::not_connected) == ENOTCONN, "not_connected");
static_assert(static_cast<int>(std::errc::not_enough_memory) == ENOMEM, "not_enough_memory");
static_assert(static_cast<int>(std::errc::not_supported) == ENOTSUP, "not_supported");
static_assert(static_cast<int>(std::errc::operation_canceled) == ECANCELED, "operation_canceled");
static_assert(static_cast<int>(std::errc::operation_in_progress) == EINPROGRESS, "operation_in_progress");
static_assert(static_cast<int>(std::errc::operation_not_permitted) == EPERM, "operation_not_permitted");
static_assert(static_cast<int>(std::errc::operation_not_supported) == EOPNOTSUPP, "operation_not_supported");
static_assert(static_cast<int>(std::errc::operation_would_block) == EWOULDBLOCK, "operation_would_block");
static_assert(static_cast<int>(std::errc::permission_denied) == EACCES, "permission_denied");
static_assert(static_cast<int>(std::errc::protocol_error) == EPROTO, "protocol_error");
static_assert(static_cast<int>(std::errc::protocol_not_supported) == EPROTONOSUPPORT, "protocol_not_supported");
static_assert(static_cast<int>(std::errc::read_only_file_system) == EROFS, "read_only_file_system");
static_assert(static_cast<int>(std::errc::resource_deadlock_would_occur) == EDEADLK, "resource_deadlock_would_occur");
static_assert(static_cast<int>(std::errc::resource_unavailable_try_again) == EAGAIN, "resource_unavailable_try_again");
static_assert(static_cast<int>(std::errc::result_out_of_range) == ERANGE, "result_out_of_range");
static_assert(static_cast<int>(std::errc::state_not_recoverable) == ENOTRECOVERABLE, "state_not_recoverable");
static_assert(static_cast<int>(std::errc::text_file_busy) == ETXTBSY, "text_file_busy");
static_assert(static_cast<int>(std::errc::timed_out) == ETIMEDOUT, "timed_out");
static_assert(static_cast<int>(std::errc::too_many_files_open) == EMFILE, "too_many_files_open");
static_assert(static_cast<int>(std::errc::too_many_files_open_in_system) == ENFILE, "too_many_files_open_in_system");
static_assert(static_cast<int>(std::errc::too_many_links) == EMLINK, "too_many_links");
static_assert(static_cast<int>(std::errc::too_many_symbolic_link_levels) == ELOOP, "too_many_symbolic_link_levels");
static_assert(static_cast<int>(std::errc::value_too_large) == EOVERFLOW, "value_too_large");
static_assert(static_cast<int>(std::errc::wrong_protocol_type) == EPROTOTYPE, "wrong_protocol_type");

int main()
{
  std::errc e = std::errc::value_too_large;
  assert(static_cast<int>(e) == EOVERFLOW);
  return 0;
}
