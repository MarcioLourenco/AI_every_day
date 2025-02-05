provider "aws" {
  region = "us-east-1"  # Altere para a região de sua preferência
}

resource "aws_s3_bucket" "meu_bucket" {
  bucket = "meu-bucket-ml"  # Altere o nome do bucket (deve ser único globalmente)

  tags = {
    Environment = "Production"
  }
}

resource "aws_s3_bucket_public_access_block" "block_public_access" {
  bucket                  = aws_s3_bucket.meu_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}