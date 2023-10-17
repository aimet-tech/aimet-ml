import os


def get_commit_id(short: bool = True) -> str:
    """
    Get the Git commit ID of the current repository.

    Args:
        short (bool, optional): Whether to get a short or full Git commit ID. Defaults to True.

    Returns:
        str: The Git commit ID as a string.
    """
    if short:
        git_command = "git rev-parse --short HEAD"
    else:
        git_command = "git rev-parse HEAD"

    commit_id = os.popen(git_command).read().strip()

    return commit_id
