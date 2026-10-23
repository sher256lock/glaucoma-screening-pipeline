# Contributing to Glaucoma Binary Screening Pipeline

Thank you for contributing to the Glaucoma Screening prototype! To ensure quality and consistency across releases, please follow these guidelines.

---

## 1. Branching Strategy (GitHub Flow)

We use **GitHub Flow** for ongoing feature work:

- `main`: Protected, production-ready, passing all CI tests.
- Feature branches: Created from `main`, named with clear prefixes:
  - `feat/<feature-name>`: New capabilities (e.g., `feat/external-eval-harness`)
  - `fix/<bug-name>`: Bug fixes (e.g., `fix/api-health-head-method`)
  - `docs/<doc-name>`: Documentation updates (e.g., `docs/update-architecture`)
  - `test/<test-name>`: Test improvements (e.g., `test/e2e-pdf-streaming`)

---

## 2. Commit Message Conventions

We enforce [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description in imperative mood>

[optional body explaining why, not just what]
```

**Common Types:**
- `feat`: New feature or capability
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring without behavior changes
- `perf`: Performance or latency optimizations
- `chore`: Tooling, dependencies, or git maintenance

**Example:**
```bash
git commit -m "feat(api): add HEAD method support to /health endpoint"
```

---

## 3. Local Verification Before Opening a PR

Before opening a pull request, verify that all tests pass locally:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Ensure:
1. All 57+ unit and integration tests pass cleanly.
2. In-memory PHI constraints are respected (never write uploaded patient fundus images to disk).
3. Terminology constraint: Output scores are strictly labeled **"Glaucoma Suspicion Score"** (never "probability" or "confidence").

---

## 4. Submitting a Pull Request

1. Push your branch to GitHub:
   ```bash
   git push -u origin feat/your-feature-name
   ```
2. Open a Pull Request on GitHub against `main`.
3. Complete all fields in the Pull Request template.
4. Ensure CI tests pass.
