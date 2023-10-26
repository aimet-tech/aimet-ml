import pandas as pd

import wandb


def table_to_dataframe(table: wandb.Table) -> pd.DataFrame:
    """
    Convert a WandB table to a Pandas DataFrame.

    Args:
        table (wandb.Table): The WandB table to be converted.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the WandB table.
    """
    return pd.DataFrame(data=table.data, columns=table.columns)


def list_artifact_names(
    api: wandb.Api, artifact_type: str, with_versions: bool = True, with_aliases: bool = True, per_page: int = 100
) -> list:
    """
    List available artifact names for a specific artifact type.

    Args:
        api (wandb.Api): The WandB API client.
        artifact_type (str): The type of artifact for which names are listed.
        with_versions (bool, optional): Include version suffixes. Defaults to True.
        with_aliases (bool, optional): Include artifact aliases. Defaults to True.
        per_page (int, optional): Number of items to retrieve per page. Defaults to 100.

    Returns:
        list: A sorted list of available artifact names with optional suffixes (versions or aliases).
    """
    available_artifact_names = set()

    for collection in api.artifact_type(artifact_type).collections():
        suffixes = set()

        if with_aliases:
            suffixes.update(collection.aliases)

        if with_versions:
            suffixes.update([v.version for v in collection.versions(per_page=per_page)])

        if suffixes:
            available_artifact_names.update([f"{collection.name}:{suffix}" for suffix in suffixes])
        else:
            available_artifact_names.update([collection.name])

    return sorted(available_artifact_names)


def load_artifact(
    api: wandb.Api, artifact_type: str, artifact_name: str, artifact_alias: str, per_page: int = 100
) -> wandb.Artifact:
    """
    Load a WandB artifact by name and alias.

    Args:
        api (wandb.Api): The WandB API client.
        artifact_type (str): The type of artifact to load.
        artifact_name (str): The base name of the artifact.
        artifact_alias (str): The alias of the artifact.
        per_page (int, optional): Number of items to retrieve per page. Defaults to 100.

    Returns:
        wandb.Artifact: The loaded WandB artifact or None if it doesn't exist.
    """
    available_artifact_types = [t.name for t in api.artifact_types()]
    if artifact_type not in available_artifact_types:
        return None

    available_artifact_names = list_artifact_names(api, artifact_type, per_page=per_page)

    artifact_name_with_alias = f"{artifact_name}:{artifact_alias}"

    if artifact_name_with_alias not in available_artifact_names:
        return None

    return wandb.use_artifact(artifact_name_with_alias)
