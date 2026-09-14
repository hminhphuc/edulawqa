# Frozen measurement checksums

The scoring rubric was frozen before any system comparison and never changed afterwards.
`harness/judge.py` refuses to run if the configuration file no longer matches this
checksum, so that a silently edited rubric cannot reach a reported number.

| File | SHA-256 |
|---|---|
| `configs/judge.frozen.yaml` | `a2c99278f25738a5a4f9c163ccd5a39eee56feb79110446b2adde7e2a364df8f` |

Verify:

```bash
sha256sum configs/judge.frozen.yaml
```

Released benchmark checksums are in `data/SHA256SUMS`.
