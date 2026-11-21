Uncommitted work from the removed sd.cpp worktrees (deleted for disk space 2026-11-19, Greg's OK).
The branches `h3-image` (8406b85) and `moss-merge` (22f4fc6) are still in engines/sd.cpp.
Restore: `git -C engines/sd.cpp worktree add ../sd.cpp-h3img h3-image && git -C engines/sd.cpp-h3img apply ../archive/sd.cpp-h3img_h3-image_wip.patch`
(the moss ggml patch applies inside its ggml submodule at the commit in sd.cpp-moss_ggml_base.txt).
