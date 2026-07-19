variable "aws_profile" {
  description = <<-EOT
    AWS profile to use — the one backed by your SSO session for the target account. Account-specific, so
    it has no default: set it in a local terraform.tfvars (gitignored) or via TF_VAR_aws_profile. Keeping
    it out of the committed files means nothing here identifies the AWS account.
  EOT
  type        = string
}

variable "region" {
  description = "Region to deploy into. One region, one machine — the whole point is a fixed host."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Tag/name prefix on every resource, so this stack is self-identifying and isolatable."
  type        = string
  default     = "nautilus-bench-runner"
}

variable "github_repo" {
  description = "owner/repo the runner registers with."
  type        = string
  default     = "geospatial-jeff/nautilus"
}

variable "runner_labels" {
  description = "Comma-separated runner labels; the bench workflows target 'nautilus-bench'."
  type        = string
  default     = "nautilus-bench"
}

variable "runner_name" {
  description = "The runner's display name in the repo's Settings -> Actions -> Runners."
  type        = string
  default     = "nautilus-bench-aws"
}

variable "github_runner_token" {
  description = <<-EOT
    Short-lived GitHub Actions RUNNER REGISTRATION token (expires ~1 hour). Never commit it and never put
    it in a .tfvars file — pass it at apply time. Mint one with:
      gh api -X POST repos/OWNER/REPO/actions/runners/registration-token -q .token
    See README.md.
  EOT
  type        = string
  sensitive   = true
}

variable "instance_type" {
  description = "c7i.large = 2 vCPU Sapphire Rapids: one core runs the benchmark, one absorbs OS/runner."
  type        = string
  default     = "c7i.large"
}

variable "root_volume_gb" {
  description = "Root gp3 volume size (GiB)."
  type        = number
  default     = 30
}

variable "vpc_cidr" {
  description = "CIDR for the dedicated, un-peered VPC. Any private range works since nothing peers to it."
  type        = string
  default     = "10.42.0.0/16"
}
