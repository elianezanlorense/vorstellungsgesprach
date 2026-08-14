#!/usr/bin/env bash
set -euo pipefail

BASHRC="$HOME/.bashrc"

# Adiciona uma configuração ao ~/.bashrc somente se ela ainda não existir
add_to_bashrc() {
    local line="$1"

    if ! grep -Fqx "$line" "$BASHRC"; then
        echo "$line" >> "$BASHRC"
    fi
}

# Configura o prompt
add_to_bashrc 'PS1="> "'

# Configura os atalhos do Git
add_to_bashrc "alias st='git status'"
add_to_bashrc "alias sw='git switch'"
add_to_bashrc "alias br='git branch'"
add_to_bashrc "alias co='git checkout'"
add_to_bashrc "alias cm='git commit'"
add_to_bashrc "alias ps='git push'"
add_to_bashrc "alias pl='git pull'"
add_to_bashrc "alias ga='git add'"
add_to_bashrc "alias lg='git log --oneline --graph --decorate --all'"

# Verifica se o nome da branch foi informado
if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <nome-da-branch>"
    echo "Exemplo: $0 dev"
    exit 1
fi

BRANCH="$1"

# Verifica se estamos dentro de um repositório Git
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Erro: execute este script dentro de um repositório Git."
    exit 1
fi

# Atualiza a branch main
git switch main
git pull --ff-only origin main

# Muda para a branch ou cria uma nova
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "A branch '$BRANCH' já existe. Mudando para ela..."
    git switch "$BRANCH"
else
    echo "Criando a branch '$BRANCH'..."
    git switch -c "$BRANCH"
fi

# Instala o uv caso ainda não esteja instalado
if ! command -v uv >/dev/null 2>&1; then
    echo "Instalando o uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Inicializa o projeto Python somente quando necessário
if [[ ! -f pyproject.toml ]]; then
    uv init
fi

# Cria/sincroniza o ambiente virtual
export UV_LINK_MODE=copy
uv sync

echo
echo "Projeto configurado com sucesso."
echo "Branch atual: $(git branch --show-current)"
echo
echo "Para atualizar o terminal atual, execute:"
echo "source ~/.bashrc"
echo
echo "Para ativar o ambiente virtual, execute:"
echo "source .venv/bin/activate"