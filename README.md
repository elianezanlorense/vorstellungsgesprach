# vorstellungsgesprach


bash start-project.sh dev
source ~/.bashrc
alias | grep git
source .venv/bin/activate

git remote -v
git log --oneline -3

uv add python-frontmatter python-dotenv 
uv add lingua-language-detector

python3 -m json.tool job.json > /dev/null && echo "JSON válido" || echo "JSON com problema"
python3 -c "import json; print(len(json.load(open('data/raw/job.json'))))"
