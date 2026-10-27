"""Experiment lineage and initialization policies, independent of cloud provisioning."""


def resolve_branch_source(c):
    from .branch import resolve_branch_source as resolve
    return resolve(c)


def import_shared_prefix(root, c):
    from .branch import import_shared_prefix as restore
    return restore(root, c)
