"""
Sobe todos os arquivos da pasta data/raw/ para um dataset no Hugging Face Hub.

Zero configuracao manual necessaria alem do token:
- O nome de usuario do Hugging Face e descoberto automaticamente a partir do token.
- O nome do repositorio de dataset e derivado do nome do repositorio GitHub atual
  (variavel GITHUB_REPOSITORY, ja fornecida automaticamente pelo GitHub Actions).
- O dataset e criado automaticamente no Hugging Face se ainda nao existir.

Requer apenas a variavel de ambiente HF_TOKEN (token com permissao de escrita).
No GitHub Actions, isso vem de secrets.HF_TOKEN.
Localmente, voce pode rodar: export HF_TOKEN=seu_token_aqui
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_TYPE = "dataset"
LOCAL_FOLDER = "data/raw"
PATH_IN_REPO = "data/raw"


def resolve_repo_id(api: HfApi) -> str:
    """Descobre automaticamente o repo_id no formato usuario/nome-do-repo."""
    manual_override = os.environ.get("HF_REPO_ID")
    if manual_override:
        return manual_override

    whoami = api.whoami()
    hf_username = whoami["name"]

    github_repository = os.environ.get("GITHUB_REPOSITORY")  # ex: usuario/nome-do-repo
    if github_repository:
        repo_name = github_repository.split("/")[-1]
    else:
        repo_name = Path.cwd().name

    return f"{hf_username}/{repo_name}"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Variavel de ambiente HF_TOKEN nao encontrada.")

    local_path = Path(LOCAL_FOLDER)
    if not local_path.exists():
        raise FileNotFoundError(f"Pasta {LOCAL_FOLDER} nao encontrada.")

    api = HfApi(token=token)
    repo_id = resolve_repo_id(api)

    print(f"Dataset alvo detectado automaticamente: {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, exist_ok=True, private=False)

    print(f"Enviando arquivos de '{LOCAL_FOLDER}' para '{repo_id}'...")

    api.upload_folder(
        folder_path=str(local_path),
        path_in_repo=PATH_IN_REPO,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message="Atualiza dataset de vagas de TI",
    )

    print("Upload concluido com sucesso.")


if __name__ == "__main__":
    main()
