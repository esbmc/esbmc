from jira_unavailable_pkg import JIRA


def connect() -> int:
    client = JIRA()
    return 0


connect()
