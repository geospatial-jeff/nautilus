data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runner" {
  name               = "${var.project}-role"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

# The only capability the instance gets: SSM Session Manager, so you can open a shell on it without SSH,
# a key pair, or any inbound port. Nothing here grants access to other account resources.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.runner.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "runner" {
  name = "${var.project}-profile"
  role = aws_iam_role.runner.name
}
