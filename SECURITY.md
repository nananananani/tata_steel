# Security Notice

## ⚠️ IMPORTANT: Credential Rotation Required

**Date**: 2026-01-29

### Issue
Previous commits (`5a34460` and earlier) contained hardcoded Cloudinary API credentials in `cloudinary_upscale.py`.

### Immediate Actions Required

1. **Rotate your Cloudinary credentials immediately**:
   - Go to https://cloudinary.com/console/settings/security
   - Generate new API keys
   - Update your local `.env` file with the new credentials
   - Delete the old credentials from Cloudinary dashboard

2. **Verify** that `.env` is in `.gitignore` (✅ Done as of commit `06015f1`)

### Git History

⚠️ **Note**: The exposed credentials are still in git history. If this is a security concern:

**Option 1: Soft approach** (Rotate credentials only)
- Simply use the new credentials from `.env` going forward
- Old keys in history are now invalid

**Option 2: Hard approach** (Clean git history) ⚠️ Destructive!
```bash
# WARNING: This rewrites history and requires force push
# Only do this if absolutely necessary
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch cloudinary_upscale.py" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (DANGER: breaks clones for collaborators)
git push origin --force --all
```

### Current Security Status (as of commit `06015f1`)

✅ **Secured**:
- `.env` added to `.gitignore`
- `.env.example` template created (no secrets)
- `cloudinary_upscale.py` now uses `os.getenv()`
- README updated with setup instructions

❌ **Still in git history** (commits before `06015f1`):
- Cloudinary credentials in `cloudinary_upscale.py`

### Best Practices Going Forward

1. ✅ Never commit `.env` files
2. ✅ Always use environment variables for secrets
3. ✅ Rotate credentials if accidentally exposed
4. ✅ Use `.env.example` as a template
5. ✅ Run `git log -S "api_key"` before pushing to check for secrets

---
**Last Updated**: 2026-01-29
