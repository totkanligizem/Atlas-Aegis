# Data Retention and Access Policy (Draft)

## Roles
- Owner: approves source onboarding and release.
- Reviewer: validates legal and privacy controls.
- Operator: runs pipeline jobs with least-privilege access.

## Retention Windows
- Raw temporary collection clips: 7 days.
- Labeled training subset: 90 days (review every 30 days).
- Final release validation set: 180 days.
- Derived reports/metrics: 180 days.

## Access Rules
- Principle of least privilege.
- Production data writable only by pipeline service account.
- Human access requires owner approval and audit trail.

## Deletion Rules
- Time-based deletion at retention expiry.
- Immediate deletion when permission is revoked.
- Deletion events logged with timestamp and approver.
