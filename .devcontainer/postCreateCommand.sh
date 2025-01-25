if [ -x "$(command -v git)" ]
then
  git config --global init.defaultBranch main
  git config --global --add safe.directory "*"
  git config --global user.name "$GIT_USER_NAME"
  git config --global user.email "$GIT_USER_EMAIL"
fi

if [ -x "$(command -v gh)" ]
then
  gh auth setup-git
fi

if [ -x "$(command -v huggingface-cli)" ] && [ -v "$HF_TOKEN" ]
then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

conda env export > environment.yml
pip list --format=freeze > requirements_freeze.txt
