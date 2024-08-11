variable "aws_region" {
  description = "The AWS region to create resources in."
  default     = "us-east-1"
}

variable "s3_bucket_name" {
  description = "The name of the S3 bucket."
  default     = "flask-app-model-bucket"
}

variable "ami_id" {
  description = "The AMI ID for the EC2 instance."
  default     = "ami-0a0e5d9c7acc336f1" # Update with a valid AMI ID
}

variable "instance_type" {
  description = "The type of EC2 instance."
  default     = "t2.micro"
}

variable "vpc_id" {
  description = "VPC ID to set the traffic"
  default     = "vpc-0a6d9a9932ee1c207"
}

#vpc-0a6d9a9932ee1c207 
