# Latest Amazon Linux 2023 x86_64, resolved from the AWS-published SSM parameter (no hardcoded AMI id).
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_instance" "runner" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.runner.id
  vpc_security_group_ids = [aws_security_group.runner.id]
  iam_instance_profile   = aws_iam_instance_profile.runner.name

  # Installs the GitHub Actions runner and registers it with the repo (see userdata.sh.tftpl).
  user_data = templatefile("${path.module}/userdata.sh.tftpl", {
    github_repo   = var.github_repo
    runner_token  = var.github_runner_token
    runner_labels = var.runner_labels
    runner_name   = var.runner_name
  })
  # A changed registration token means "re-provision the runner", so replace the instance rather than
  # silently ignoring the new user_data.
  user_data_replace_on_change = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required" # IMDSv2 only — blocks the SSRF-to-credentials class of attack
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
    encrypted   = true
  }

  tags = { Name = var.project }
}
