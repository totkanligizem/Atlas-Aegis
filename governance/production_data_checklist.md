# Production Data Governance Checklist

## Acquisition
- [ ] Data source owner identified and documented.
- [ ] Written permission for operational use archived.
- [ ] Geographic and site restrictions documented.
- [ ] Collection date/time and device metadata recorded.

## Privacy and Safety
- [ ] Faces/plates/personally identifiable signals reviewed.
- [ ] Required anonymization policy applied (blur/mask/redact).
- [ ] Safety-sensitive zones excluded from collection plan.

## Labeling and QA
- [ ] Label taxonomy frozen and versioned.
- [ ] Labeler guideline document versioned.
- [ ] Double-pass QA on at least 10% of clips.
- [ ] Disagreement resolution log archived.

## Access and Retention
- [ ] Access control list defined (owner, reviewer, operator).
- [ ] Retention period defined and approved.
- [ ] Deletion process and audit log in place.
- [ ] Backup policy defined for final artifacts.

## Release Gate
- [ ] `governance/legal_status.yaml` reviewed and signed.
- [ ] Model card contains dataset provenance table.
- [ ] R&D and production pipelines are physically/logically separated.
