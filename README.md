# Vishnu AI Agent Deployment Guide

## Prerequisites
- AWS Account (Free Tier)
- Node.js and npm (for Serverless Framework)
- Python 3.9+
- AWS CLI configured with credentials
- Ghostscript installed on the Lambda environment (use a custom layer if needed)

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd project


Install Dependencies
pip install -r requirements.txt
npm install -g serverless


Configure Environment Variables

Create a .env file in the root directory.
Add your AWS S3 bucket name, AWS credentials, Google API key, and OpenAI API key as shown in .env template.
Ensure the S3 bucket exists in the ap-south-1 region.


Deploy to AWS Lambda
serverless deploy

This deploys the FastAPI app to Lambda via API Gateway. Note the endpoint URL provided after deployment.

Set Up AWS Budget Alert

Log in to the AWS Management Console.
Navigate to AWS Budgets > Create Budget.
Select "Cost budget", set monthly budget to $5.
Configure an alert threshold at 100% ($5).
Add an email notification (requires SNS topic).
Save the budget.


Test the Application

Open the API Gateway endpoint in a browser to access the frontend.
Test chatbot and PDF operations (merge, compress, encrypt, convert to images).



Notes

Free Tier Limits: The deployment uses 256MB Lambda functions, well within the 1M requests and 3.2M seconds/month Free Tier. S3 usage is minimal (Free Tier: 5GB storage, 20,000 GET, 2,000 PUT requests).
Memory Optimization: Files are stored in S3 during processing, and temporary files are cleaned up after each operation. Lambda's stateless nature avoids memory exhaustion for 15-20 concurrent users.
Ghostscript: Ensure Ghostscript is available in the Lambda environment. You may need to create a Lambda layer with Ghostscript binaries.
Scaling: Lambda automatically scales to handle concurrent users. If you exceed Free Tier limits, the budget alert will notify you at $5.

Troubleshooting

Memory Issues: Monitor Lambda logs in CloudWatch for memory errors. Reduce DPI for compression or limit file sizes if needed.
S3 Permissions: Ensure the Lambda role has s3:PutObject, s3:GetObject, and s3:DeleteObject permissions.
API Errors: Check API Gateway logs for CORS or timeout issues.

For issues, contact AWS Support or check the FastAPI and Serverless Framework documentation.```
