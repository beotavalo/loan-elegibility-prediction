provider "aws" {
  region = var.aws_region
}

# Crear un grupo de seguridad que permita HTTP y HTTPS
resource "aws_security_group" "allow_http_https" {
  name        = "allow_http_https"
  description = "Allow HTTP and HTTPS traffic"
  vpc_id      = var.vpc_id  # Asegúrate de tener el ID de tu VPC

  # Permitir tráfico HTTP
  ingress {
    description      = "Allow HTTP traffic"
    from_port        = 80
    to_port          = 80
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
  }

  # Permitir tráfico HTTPS
  ingress {
    description      = "Allow HTTPS traffic"
    from_port        = 443
    to_port          = 443
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
  }

  # Permitir todo el tráfico de salida
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
  }

  tags = {
    Name = "allow_http_https"
  }
}

resource "aws_instance" "flask_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type
  # Asociar el grupo de seguridad creado a la instancia
  vpc_security_group_ids = [aws_security_group.allow_http_https.id]

  user_data = <<-EOF
              #!/bin/bash
              # Update packages list
              sudo apt-get update -y

              #Install docker on EC2
              sudo apt-get install -y docker.io

              #Star and enable docker
              sudo systemctl start docker
              sudo systemctl enable docker              
              
              #Pull the image from docker hub
              sudo docker pull botavalo/flask-app:latest
              
              #Run the image
              sudo sudo docker run -d -p 80:5000 28d69136c4b2
              
              EOF

  tags = {
    Name = "FlaskAppInstance"
  }
}



output "instance_ip" {
  value = aws_instance.flask_instance.public_ip
}
