provider "aws" {
  region  = var.region
  profile = var.aws_profile

  # Every resource this stack creates is tagged, so it is trivially identifiable and isolatable in a
  # shared account — nothing here touches or depends on any pre-existing resource.
  default_tags {
    tags = {
      Project   = var.project
      Component = "github-actions-runner"
      Repo      = var.github_repo
      ManagedBy = "terraform"
    }
  }
}
