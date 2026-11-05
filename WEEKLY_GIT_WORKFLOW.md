# Weekly Git Workflow: Branch, Commit, Merge to Main

Project: Hybrid ML-Based IDS (SKIT-IOT-2023-2027-F-031)
Owner: Vivek (API, Security, Frontend & Backend)
Repo: https://github.com/Vvivekvyas/SKIT-IOT-2023-2027-F-031
Local folder: `D:\Final year project\SKIT-IOT-2023-2027-F-031`

---

## 1. Progress so far

| Week | Task | Branch | Status |
|---|---|---|---|
| 7 | API validation + error handling | `vivek/week-7-validation-errors` | Merged to main |
| 8 | Backend integration for dataset endpoints | `vivek/week-8-dataset-integration` | Merged to main |
| 9 | Prediction API prototype (secured) | `vivek/week-9-prediction-api` | Code written and copied in. Commit, push and merge still to do (see section 4) |
| 10 | Secure model-service interface | not started | |

### What Week 9 adds

- `POST /api/v1/predict`: classifies one network flow. Needs a JWT with analyst role or higher. Rate limited.
- `GET /api/v1/predict/model`: shows whether a model is loaded and which features it expects.
- `app/services/predictor.py`: loads the ML team's hybrid model once at startup.
- Demo mode (`PREDICT_DEMO_MODE=true` in `backend\.env`): runs the real model code with untrained random weights. Responses say `"model_version": "demo-untrained"`. Off by default. Ignored in production.
- A new `anomaly_score` column in the `predictions` table. It is added automatically on startup.
- `torch` added to `requirements.txt`.
- With no model and demo mode off, `/predict` returns 503. It never returns fake predictions.

---

## 2. The golden rules

1. **Always run git commands from the repo root.** The prompt must end in `SKIT-IOT-2023-2027-F-031>`, not `backend>` and not `Final year project>`.
2. **One branch per week**, named `vivek/week-N-short-name`.
3. **Run `git status` before every commit.** It must never list `.env`, `.db` files, `uploads/` or `.pyc` files.
4. **Read the output of each command before running the next one.** If anything looks different from what is described here, stop and ask.
5. **`Already up to date` or `Everything up-to-date` after a commit, push or merge means nothing was sent.** It usually means the commit never happened.

---

## 3. The weekly routine, step by step

### Step 1: Start from a clean main (terminal)

```powershell
cd "D:\Final year project\SKIT-IOT-2023-2027-F-031"
git status
git checkout main
git pull origin main
git log --oneline -5
```

- `git status` should say `nothing to commit, working tree clean`.
- Check that last week's commit or merge is in the log.
- If there is an open pull request on GitHub for last week's work, merge it first.

### Step 2: Create this week's branch (terminal)

```powershell
git checkout -b vivek/week-9-prediction-api
```

It should say `Switched to a new branch 'vivek/week-9-prediction-api'`.

### Step 3: Copy the new files in (File Explorer)

1. Extract the weekly zip.
2. Copy the folders inside it (for example `backend`) into the repo root.
3. When Windows asks about files with the same name, choose **Replace the files in the destination**.

### Step 4: Install anything new (terminal)

Only if the week adds packages:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch
```

### Step 5: Run and test (terminal, then browser)

```powershell
cd backend
uvicorn app.main:app --reload
```

- Open http://127.0.0.1:8000/docs
- Log in as `analyst@example.com`, click **Authorize**, and try the new endpoints.
- Take screenshots for your weekly proof.
- Press **Ctrl + C** to stop the server.

### Step 6: Go back to the repo root (terminal)

```powershell
cd ..
```

Run this **once**. Check that the prompt ends in `SKIT-IOT-2023-2027-F-031>`.

### Step 7: Check what will be committed (terminal)

```powershell
git status
```

Expected: only `backend/...` files (and `docs/...` if you added docs). No `.env`, `.db`, `uploads/` or `.pyc`.

### Step 8: Stage, commit and push the branch (terminal)

```powershell
git add backend
git commit -m "Week 9: secured prediction API prototype"
git push -u origin vivek/week-9-prediction-api
```

Expected output:

- `git add`: no output (a line-ending warning about LF and CRLF is harmless).
- `git commit`: `N files changed` plus `create mode` lines for new files.
- `git push`: uploads objects and prints `old..new  branch -> branch`.

### Step 9: Merge into main (terminal)

```powershell
git checkout main
git pull origin main
git merge --no-edit vivek/week-9-prediction-api
git push origin main
```

Expected:

- The merge prints `Fast-forward` or `Merge made by the 'ort' strategy`, followed by a list of files.
- The final push prints `main -> main`.

### Step 10: Verify (terminal)

```powershell
git log --oneline -3
git ls-files backend/app/api/v1/predict.py backend/app/services/predictor.py
```

Your week's commit should be near the top of the log. The file command should print both paths.

---

## 4. Week 9 checklist (where you are now)

- [x] Week 8 merged into main
- [x] Branch `vivek/week-9-prediction-api` created
- [x] Week 9 files copied into `backend`
- [x] `pip install torch` finished successfully
- [ ] `PREDICT_DEMO_MODE=true` added to `backend\.env`
- [ ] Server started and the `DEMO MODE` log line seen
- [ ] `/docs` checks done and screenshots taken
- [ ] `git add backend` and `git status` showing "Changes to be committed"
- [ ] `git commit` showing `8 files changed`
- [ ] `git push -u origin vivek/week-9-prediction-api` uploading objects
- [ ] Merged into main and pushed
- [ ] Verified with `git log` and `git ls-files`

### Week 9 test checks in /docs

1. `GET /api/v1/predict/model` returns `"ready": true`.
2. `POST /api/v1/predict` with all 10 features (`feature_1` to `feature_10`) returns 200.
3. The same request without `feature_10` returns 422.
4. A request with no token returns 401.

Sample body:

```json
{"features": {"feature_1": 0.1, "feature_2": 0.5, "feature_3": 1.0, "feature_4": 0.2, "feature_5": 0.0, "feature_6": 0.3, "feature_7": 0.9, "feature_8": 0.4, "feature_9": 0.6, "feature_10": 0.7}}
```

---

## 5. Mistakes we hit and how to avoid them

| What happened | Why | Fix |
|---|---|---|
| `fatal: pathspec 'backend' did not match any files` | The terminal was inside `backend`, so `git add backend` looked for `backend\backend` | `cd` to the repo root first |
| `git commit` said `no changes added to commit` | `git add` failed or was skipped, so there was nothing staged | Run `git add backend`, then check `git status` shows "Changes to be committed" |
| Push said `Everything up-to-date` | Nothing was committed, so there was nothing to send | Commit first, then push |
| Merge said `Already up to date` | The branch had no new commits | Go back to the branch and commit properly |
| `fatal: not a git repository` | An extra `cd ..` moved the terminal out of the repo to `Final year project` | `cd "D:\Final year project\SKIT-IOT-2023-2027-F-031"` |
| `git add backend docs` failed | The `docs` folder did not exist | Add only the folders that exist |
| The `M backend/...` lines when switching branches | Uncommitted edits travel with you between branches. This is normal | Commit them on the right branch |

---

## 6. If something goes wrong

- **`CONFLICT` during a merge:** stop. Do not run anything else. Copy the message and ask for help.
- **`git checkout main` refuses because of `ids_dev.db`:** copy the message and ask for the exact fix.
- **Committed the wrong files:** do not push. Copy the output of `git status` and `git log --oneline -3` and ask.
- **Never commit `.env`.** It contains secrets.

---

## 7. Open items

- Agree with Nancy where the trained model checkpoint will be saved and which scaler was used in preprocessing. The API needs both to serve real predictions. Until then `/predict` returns 503, or demo results in demo mode.
- Successful responses are still plain JSON. Only errors use the `{data, meta, error}` format. Decide with the dashboard owner whether to wrap success responses too.
- Next week (Week 10): secure model-service interface.
