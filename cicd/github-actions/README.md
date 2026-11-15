# GitHub Actions workflow archive

The comprehensive CI/CD workflow suite (`ci.full.yml`, `cd.yml`, `docs.yml`,
`ml.yml`, `release.yml`, `security.yml`) lives here as **reference** and is not
active. Only `.github/workflows/ci.yml` (the minimal smoke CI) is active so the
repository can be pushed with a Personal Access Token that lacks the `workflow`
scope.

## Activating the full suite

To activate these workflows, a token with the `workflow` scope is required:

1. GitHub → Settings → Developer settings → Personal access tokens → Generate
   new token (classic) → tick **`workflow`** (and `repo`).
2. Move each file from `cicd/github-actions/` back to `.github/workflows/`
   (rename `ci.full.yml` → `ci.yml`, overwriting the minimal one).
3. Commit and push.

Once active, the suite covers: lint/type/test matrix, pre-commit, security
scanning, ML training/eval, documentation build, release, and continuous
deployment.
