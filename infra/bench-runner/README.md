# nautilus benchmark runner (AWS)

Terraform for a single, always-on EC2 instance that acts as a **self-hosted GitHub Actions runner** for
`geospatial-jeff/nautilus`, labelled `nautilus-bench`. Its only job is to give the throughput benchmark
gate a **fixed machine**, so run-to-run variance drops from ~20% (GitHub's shared fleet) to a few percent
and the gate can actually catch real regressions.

Kept **local and out of the nautilus repo** on purpose — review before it goes anywhere.

## What it builds (all in the target account, us-east-1)
- A **dedicated VPC** + public subnet + IGW — isolated, un-peered; nothing shared with the rest of the account.
- A **security group with no inbound** — access is via **SSM Session Manager** (no SSH, no key pair, no open port).
- An **IAM role** whose only permission is SSM Session Manager.
- A **`c7i.large`** instance (Amazon Linux 2023, IMDSv2-only, encrypted gp3 root) whose user-data installs
  and registers the runner as a systemd service running as `ec2-user`.

Tear it all down with one `terraform destroy`; the account is then exactly as it was.

## Prerequisites
```bash
aws sso login --sso-session <your-sso-session>   # log into the target account
terraform version                             # >= 1.5
gh auth status                                # gh CLI, logged into the repo
```

## Deploy
The runner registration token is short-lived (~1 hour) and must never be committed, so mint it at apply time:

```bash
cd nautilus-bench-runner
terraform init

# plan for review — creates nothing
TF_VAR_github_runner_token=$(gh api -X POST repos/geospatial-jeff/nautilus/actions/runners/registration-token -q .token) \
  terraform plan

# apply once you're happy
TF_VAR_github_runner_token=$(gh api -X POST repos/geospatial-jeff/nautilus/actions/runners/registration-token -q .token) \
  terraform apply
```

Within a minute or two the runner appears **online** under repo Settings → Actions → Runners, and
`terraform output ssm_shell` prints the command to open a shell on it.

## Verify / debug
```bash
gh api repos/geospatial-jeff/nautilus/actions/runners -q '.runners[] | {name, status, labels:[.labels[].name]}'
$(terraform output -raw ssm_shell)           # shell onto the box
sudo journalctl -u 'actions.runner.*' -f     # watch the runner service
```

## Destroy
```bash
terraform destroy
# The instance is gone, but GitHub still lists an OFFLINE runner — remove it:
gh api repos/geospatial-jeff/nautilus/actions/runners --jq '.runners[] | select(.name=="nautilus-bench-aws") | .id' \
  | xargs -I{} gh api -X DELETE repos/geospatial-jeff/nautilus/actions/runners/{}
```

## Cost
`c7i.large` on-demand ≈ **$65/mo** always-on + ~$2/mo for the 30 GB gp3 root. Stopping the instance when
idle drops compute to pennies (you keep paying the small EBS), at the cost of the runner being offline
(jobs queue until it starts). Always-on was the chosen trade-off.

## Notes / possible upgrades
- **Token lifetime.** The registration token only needs to be valid during `apply`; the running runner
  re-authenticates on its own afterward. You only need a fresh token when the instance is **recreated**
  (e.g. a changed `user_data`, which `user_data_replace_on_change` forces).
- **Hands-off recreation.** To avoid supplying a token on every recreate, store a GitHub PAT in SSM
  Parameter Store (SecureString) and have user-data fetch a registration token itself at boot. Left out
  here to avoid a long-lived secret; add it if recreation frequency makes the manual token annoying.
- **State holds the token.** Terraform state contains `github_runner_token` in plaintext, so the state
  stays local (see `.gitignore`). Move to encrypted remote state (S3 + DynamoDB lock in the account)
  if more than one person will run this.
