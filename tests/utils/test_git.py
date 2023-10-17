import os

import pytest

from aimet_ml.utils.git import get_commit_id


@pytest.fixture
def in_git_repository():
    # Check if we are in a git repository
    in_repo = os.system("git rev-parse --is-inside-work-tree") == 0
    if not in_repo:
        pytest.skip("Not in a git repository.")
    yield


def test_get_commit_id_short(in_git_repository):
    commit_id = get_commit_id(short=True)
    assert len(commit_id) == 7  # Short commit ID should be 7 characters
    assert commit_id.isalnum()


def test_get_commit_id_full(in_git_repository):
    commit_id = get_commit_id(short=False)
    assert len(commit_id) == 40  # Full commit ID should be 40 characters
    assert commit_id.isalnum()
