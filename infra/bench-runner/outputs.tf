output "instance_id" {
  description = "EC2 instance id of the runner."
  value       = aws_instance.runner.id
}

output "public_ip" {
  description = "Public IP (outbound only; nothing listens on it)."
  value       = aws_instance.runner.public_ip
}

output "ssm_shell" {
  description = "Open a shell on the runner — no SSH, no key, no inbound port."
  value       = "aws ssm start-session --target ${aws_instance.runner.id} --profile ${var.aws_profile} --region ${var.region}"
}

output "runner_service_log" {
  description = "Once connected, watch the runner service."
  value       = "sudo journalctl -u 'actions.runner.*' -f"
}
