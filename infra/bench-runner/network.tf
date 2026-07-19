# A dedicated VPC so the runner shares nothing with the rest of the account. No peering, no shared
# subnets — tear this whole stack down and the account is exactly as it was.
resource "aws_vpc" "runner" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = var.project }
}

resource "aws_internet_gateway" "runner" {
  vpc_id = aws_vpc.runner.id
  tags   = { Name = var.project }
}

# One public subnet: the instance gets a public IP for outbound to GitHub + package mirrors via the IGW,
# which is far cheaper than a NAT gateway. No inbound is ever allowed (see the security group).
resource "aws_subnet" "runner" {
  vpc_id                  = aws_vpc.runner.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, 1)
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project}-public" }
}

resource "aws_route_table" "runner" {
  vpc_id = aws_vpc.runner.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.runner.id
  }
  tags = { Name = var.project }
}

resource "aws_route_table_association" "runner" {
  subnet_id      = aws_subnet.runner.id
  route_table_id = aws_route_table.runner.id
}

# Egress only — no inbound rule at all. Access to the box is via SSM Session Manager (outbound HTTPS from
# the SSM agent), so there is no open SSH port and no key pair to manage. This matters: a self-hosted
# runner on a PUBLIC repo is an attack target, so the box exposes nothing.
resource "aws_security_group" "runner" {
  name        = "${var.project}-sg"
  description = "nautilus bench runner: egress only, no inbound (SSM for shell access)"
  vpc_id      = aws_vpc.runner.id

  egress {
    description = "all outbound (GitHub, package mirrors, SSM)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = var.project }
}
