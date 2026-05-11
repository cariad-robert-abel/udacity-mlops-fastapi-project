#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import TYPE_CHECKING

import wandb

if TYPE_CHECKING:
    from wandb.sdk.lib.paths import StrPath


wandb_artifact_file_fn = wandb.Artifact.file


def wandb_artifact_file_fix(self: 'wandb.Artifact', root: str | None = None) -> 'StrPath':
    """Fix for file systems where colon (:) cannot be used in file names, such as NTFS on Windows.

    Args:
        self: W&B Artifact instance
        root: Optional root directory to use for the file path.
    """
    root = str(root or self._default_root())
    return wandb_artifact_file_fn(self, root)


wandb.Artifact.file = wandb_artifact_file_fix
